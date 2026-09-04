import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceValidationResult {
  const FaceValidationResult(this.isValid, this.message);
  final bool isValid;
  final String message;
}

class FaceDetectionService {
  final FaceDetector _detector = FaceDetector(
    options: FaceDetectorOptions(enableLandmarks: true, enableContours: true, performanceMode: FaceDetectorMode.accurate),
  );

  Future<FaceValidationResult> validate(String path) async {
    final faces = await _detector.processImage(InputImage.fromFilePath(path));
    if (faces.isEmpty) return const FaceValidationResult(false, 'لم يتم العثور على وجه.');
    if (faces.length > 1) return const FaceValidationResult(false, 'يجب أن يظهر وجه واحد فقط.');
    final face = faces.first;
    final yaw = face.headEulerAngleY?.abs() ?? 0;
    final pitch = face.headEulerAngleX?.abs() ?? 0;
    if (yaw > 25 || pitch > 25) return const FaceValidationResult(false, 'وجّه وجهك نحو الكاميرا مباشرة.');
    return const FaceValidationResult(true, 'الوجه صالح للتحليل.');
  }

  Future<void> dispose() => _detector.close();
}
