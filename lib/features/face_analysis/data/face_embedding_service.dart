import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' show Rect;

import 'package:crypto/crypto.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

/// On-device face embedding using LibreFaceRec / AuraFace-v1.
///
/// The model accepts a 112x112 RGB face crop and returns a 512-dim
/// L2-normalized embedding. The model is downloaded once and cached locally;
/// the selfie itself never leaves the device in this service.
class FaceEmbeddingService {
  static const _modelFileName = 'librefacerec-l.onnx';
  static const _modelUrl =
      'https://huggingface.co/LibreYOLO/librefacerec-l/resolve/main/librefacerec-l.onnx';
  static const _expectedSha256 =
      'a7933ea5330113b01c9b60351d8f4c33003f145d847ac5f0e52ee2effe25c60';

  OnnxRuntime? _runtime;
  OrtSession? _session;

  Future<List<double>> embed({
    required String imagePath,
    required Face face,
  }) async {
    final session = await _getSession();
    final imageBytes = await File(imagePath).readAsBytes();
    final source = img.decodeImage(imageBytes);
    if (source == null) {
      throw StateError('تعذر قراءة الصورة.');
    }

    final crop = _cropFace(source, face.boundingBox);
    final input = _toModelTensor(crop);
    final inputName = session.inputNames.first;
    final outputName = session.outputNames.first;

    final inputTensor = await OrtValue.fromList(input, [1, 3, 112, 112]);
    Map<String, OrtValue>? outputs;
    try {
      outputs = await session.run({inputName: inputTensor});
      final output = outputs[outputName];
      if (output == null) {
        throw StateError('نموذج الوجه لم يُرجع embedding.');
      }

      final values = (await output.asFlattenedList())
          .map((value) => (value as num).toDouble())
          .toList(growable: false);
      if (values.length != 512) {
        throw StateError(
          'Embedding dimension mismatch: expected 512, got ${values.length}.',
        );
      }
      return _l2Normalize(values);
    } finally {
      await inputTensor.dispose();
      if (outputs != null) {
        for (final output in outputs.values) {
          await output.dispose();
        }
      }
    }
  }

  Future<OrtSession> _getSession() async {
    if (_session != null) return _session!;

    final modelFile = await _ensureModelFile();
    _runtime = OnnxRuntime();
    _session = await _runtime!.createSession(modelFile.path);
    return _session!;
  }

  Future<File> _ensureModelFile() async {
    final directory = await getApplicationSupportDirectory();
    final file = File('${directory.path}/$_modelFileName');

    if (await file.exists()) {
      final digest = await sha256.bind(file.openRead()).first;
      if (digest.toString() == _expectedSha256) return file;
      await file.delete();
    }

    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(_modelUrl));
      final response = await request.close();
      if (response.statusCode != HttpStatus.ok) {
        throw HttpException(
          'Model download failed with HTTP ${response.statusCode}.',
        );
      }

      final sink = file.openWrite();
      try {
        await response.pipe(sink);
      } finally {
        await sink.close();
      }

      final digest = await sha256.bind(file.openRead()).first;
      if (digest.toString() != _expectedSha256) {
        await file.delete();
        throw StateError('فشل التحقق من سلامة نموذج الوجه.');
      }
      return file;
    } finally {
      client.close(force: true);
    }
  }

  img.Image _cropFace(img.Image source, Rect box) {
    final left = box.left.floor().clamp(0, source.width - 1);
    final top = box.top.floor().clamp(0, source.height - 1);
    final right = box.right.ceil().clamp(left + 1, source.width);
    final bottom = box.bottom.ceil().clamp(top + 1, source.height);
    final width = right - left;
    final height = bottom - top;
    final size = width > height ? width : height;
    final cx = left + width / 2;
    final cy = top + height / 2;

    final cropLeft = (cx - size / 2).round().clamp(0, source.width - 1);
    final cropTop = (cy - size / 2).round().clamp(0, source.height - 1);
    final cropRight = (cropLeft + size).clamp(cropLeft + 1, source.width);
    final cropBottom = (cropTop + size).clamp(cropTop + 1, source.height);

    final square = img.copyCrop(
      source,
      x: cropLeft,
      y: cropTop,
      width: cropRight - cropLeft,
      height: cropBottom - cropTop,
    );
    return img.copyResize(square, width: 112, height: 112);
  }

  List<double> _toModelTensor(img.Image image) {
    final values = <double>[];
    // NCHW, RGB. ArcFace convention: normalize pixels to [-1, 1].
    for (var channel = 0; channel < 3; channel++) {
      for (var y = 0; y < 112; y++) {
        for (var x = 0; x < 112; x++) {
          final pixel = image.getPixel(x, y);
          final value = switch (channel) {
            0 => pixel.r,
            1 => pixel.g,
            _ => pixel.b,
          };
          values.add((value.toDouble() - 127.5) / 127.5);
        }
      }
    }
    return values;
  }

  List<double> _l2Normalize(List<double> vector) {
    final norm = math.sqrt(
      vector.fold<double>(0, (sum, value) => sum + value * value),
    );
    if (norm == 0) throw StateError('Embedding norm is zero.');
    return vector.map((value) => value / norm).toList(growable: false);
  }

  Future<void> dispose() async {
    final session = _session;
    _session = null;
    if (session != null) await session.close();
    _runtime = null;
  }
}
