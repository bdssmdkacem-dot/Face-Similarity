import 'dart:async';

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceValidationResult {
  const FaceValidationResult(this.isValid, this.message, {this.face});

  final bool isValid;
  final String message;
  final Face? face;
}

class FaceDetectionService {
  final FaceDetector _realtimeDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast,
      minFaceSize: 0.15,
    ),
  );

  final FaceDetector _accurateDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableLandmarks: true,
      enableContours: true,
      performanceMode: FaceDetectorMode.accurate,
    ),
  );

  Future<List<Face>> detectRealtime(InputImage image) async {
    return _realtimeDetector.processImage(image);
  }

  Future<FaceValidationResult> validate(String path) async {
    return detectAndValidate(path);
  }

  Future<FaceValidationResult> detectAndValidate(String path) async {
    final faces = await _accurateDetector.processImage(
      InputImage.fromFilePath(path),
    );
    return validateFaces(faces);
  }

  FaceValidationResult validateFaces(List<Face> faces) {
    if (faces.isEmpty) {
      return const FaceValidationResult(false, 'لم يتم العثور على وجه.');
    }
    if (faces.length > 1) {
      return const FaceValidationResult(false, 'يجب أن يظهر وجه واحد فقط.');
    }

    final face = faces.first;
    final yaw = face.headEulerAngleY?.abs() ?? 0;
    final pitch = face.headEulerAngleX?.abs() ?? 0;
    if (yaw > 25 || pitch > 25) {
      return const FaceValidationResult(
        false,
        'وجّه وجهك نحو الكاميرا مباشرة.',
      );
    }

    return FaceValidationResult(true, 'الوجه صالح للتحليل.', face: face);
  }

  Future<void> dispose() async {
    await _realtimeDetector.close();
    await _accurateDetector.close();
  }
}
