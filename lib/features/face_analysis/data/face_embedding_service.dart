import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' show Rect;

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

/// On-device face embedding using LibreFaceRec / AuraFace-v1.
class FaceEmbeddingService {
  static const _modelFileName = 'librefacerec-l.onnx';
  static const _modelUrl = 'https://huggingface.co/LibreYOLO/librefacerec-l/resolve/main/librefacerec-l.onnx';
  static const _expectedSha256 = 'a7933ea5330113b01c9b60351d8f4c33003f145d847ac5f0e52ee2effe25c60';
  OnnxRuntime? _runtime;
  OrtSession? _session;
  Future<List<double>> embed({required String imagePath, required Face face, void Function(String status)? onStatus}) async {
    onStatus?.call('جاري تهيئة محرك الذكاء الاصطناعي…');
    final session = await _getSession(onStatus: onStatus).timeout(const Duration(seconds: 120), onTimeout: () => throw TimeoutException('تهيئة نموذج الوجه تجاوزت المهلة.'));
    onStatus?.call('جاري تجهيز صورة الوجه…');
    final source = img.decodeImage(await File(imagePath).readAsBytes());
    if (source == null) throw StateError('تعذر قراءة الصورة.');
    final inputTensor = await OrtValue.fromList(Float32List.fromList(_toModelTensor(_cropFace(source, face.boundingBox))), [1, 3, 112, 112]);
    final inputName = session.inputNames.first;
    final outputName = session.outputNames.first;
    Map<String, OrtValue>? outputs;
    try {
      onStatus?.call('جاري تشغيل نموذج الوجه…');
      outputs = await session.run({inputName: inputTensor}).timeout(const Duration(seconds: 60), onTimeout: () => throw TimeoutException('استدلال نموذج الوجه تجاوز 60 ثانية.'));
      final output = outputs[outputName];
      if (output == null) throw StateError('نموذج الوجه لم يُرجع embedding.');
      final values = (await output.asFlattenedList()).map((v) => (v as num).toDouble()).toList(growable: false);
      if (values.length != 512) throw StateError('Embedding dimension mismatch: expected 512, got ${values.length}.');
      onStatus?.call('تم إنشاء بصمة الوجه.');
      return _l2Normalize(values);
    } finally {
      await inputTensor.dispose();
      if (outputs != null) for (final output in outputs.values) await output.dispose();
    }
  }
  Future<OrtSession> _getSession({void Function(String status)? onStatus}) async {
    if (_session != null) return _session!;
    onStatus?.call('جاري تجهيز نموذج الوجه…');
    final modelFile = await _ensureModelFile(onStatus: onStatus);
    onStatus?.call('جاري تحميل النموذج داخل الجهاز…');
    _runtime = OnnxRuntime();
    final session = await _runtime!.createSession(modelFile.path).timeout(const Duration(seconds: 90), onTimeout: () => throw TimeoutException('تحميل نموذج الوجه داخل ONNX Runtime تجاوز 90 ثانية.'));
    _session = session;
    return session;
  }
  Future<File> _ensureModelFile({void Function(String status)? onStatus}) async {
    final file = File('${(await getApplicationSupportDirectory()).path}/$_modelFileName');
    if (await file.exists()) {
      onStatus?.call('جاري التحقق من نموذج الوجه…');
      if ((await sha256.bind(file.openRead()).first).toString() == _expectedSha256) return file;
      await file.delete();
    }
    onStatus?.call('جاري تنزيل نموذج الوجه لأول مرة…');
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(_modelUrl)).timeout(const Duration(seconds: 30));
      request.headers.set(HttpHeaders.userAgentHeader, 'Shabah/0.1 Android');
      final response = await request.close().timeout(const Duration(seconds: 60));
      if (response.statusCode != HttpStatus.ok) throw HttpException('Model download failed with HTTP ${response.statusCode}.');
      final sink = file.openWrite();
      try { await for (final chunk in response) { sink.add(chunk); } } finally { await sink.close(); }
      onStatus?.call('جاري التحقق من نموذج الوجه…');
      if ((await sha256.bind(file.openRead()).first).toString() != _expectedSha256) { await file.delete(); throw StateError('فشل التحقق من سلامة نموذج الوجه.'); }
      return file;
    } finally { client.close(force: true); }
  }
  img.Image _cropFace(img.Image source, Rect box) {
    final left = box.left.floor().clamp(0, source.width - 1), top = box.top.floor().clamp(0, source.height - 1);
    final right = box.right.ceil().clamp(left + 1, source.width), bottom = box.bottom.ceil().clamp(top + 1, source.height);
    final width = right - left, height = bottom - top, size = width > height ? width : height;
    final cx = left + width / 2, cy = top + height / 2;
    final cropLeft = (cx - size / 2).round().clamp(0, source.width - 1), cropTop = (cy - size / 2).round().clamp(0, source.height - 1);
    final cropRight = (cropLeft + size).clamp(cropLeft + 1, source.width), cropBottom = (cropTop + size).clamp(cropTop + 1, source.height);
    return img.copyResize(img.copyCrop(source, x: cropLeft, y: cropTop, width: cropRight - cropLeft, height: cropBottom - cropTop), width: 112, height: 112);
  }
  List<double> _toModelTensor(img.Image image) {
    final values = <double>[];
    for (var channel = 0; channel < 3; channel++) for (var y = 0; y < 112; y++) for (var x = 0; x < 112; x++) {
      final pixel = image.getPixel(x, y);
      final value = switch (channel) { 0 => pixel.r, 1 => pixel.g, _ => pixel.b };
      values.add((value.toDouble() - 127.5) / 127.5);
    }
    return values;
  }
  List<double> _l2Normalize(List<double> vector) {
    final norm = math.sqrt(vector.fold<double>(0, (sum, value) => sum + value * value));
    if (norm == 0) throw StateError('Embedding norm is zero.');
    return vector.map((value) => value / norm).toList(growable: false);
  }
  Future<void> dispose() async { final session = _session; _session = null; if (session != null) await session.close(); _runtime = null; }
}
