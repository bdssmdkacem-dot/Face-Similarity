import 'dart:ui' show Point, Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'package:face_similarity/features/face_analysis/data/face_alignment_service.dart';

void main() {
  test('aligns five ML Kit landmarks to 112x112 ArcFace template', () {
    final landmarks = <FaceLandmarkType, FaceLandmark?>{
      FaceLandmarkType.leftEye: FaceLandmark(
        type: FaceLandmarkType.leftEye,
        position: const Point<int>(38, 52),
      ),
      FaceLandmarkType.rightEye: FaceLandmark(
        type: FaceLandmarkType.rightEye,
        position: const Point<int>(74, 52),
      ),
      FaceLandmarkType.noseBase: FaceLandmark(
        type: FaceLandmarkType.noseBase,
        position: const Point<int>(56, 72),
      ),
      FaceLandmarkType.leftMouth: FaceLandmark(
        type: FaceLandmarkType.leftMouth,
        position: const Point<int>(42, 92),
      ),
      FaceLandmarkType.rightMouth: FaceLandmark(
        type: FaceLandmarkType.rightMouth,
        position: const Point<int>(71, 92),
      ),
    };

    final face = Face(
      boundingBox: const Rect.fromLTWH(20, 30, 80, 80),
      landmarks: landmarks,
      contours: const {},
    );
    final source = img.Image(width: 112, height: 112);
    final aligned = FaceAlignmentService().align(source, face);

    expect(aligned, isNotNull);
    expect(aligned!.width, 112);
    expect(aligned.height, 112);
  });

  test('returns null when a required landmark is missing', () {
    final face = Face(
      boundingBox: const Rect.fromLTWH(20, 30, 80, 80),
      landmarks: const {},
      contours: const {},
    );
    final source = img.Image(width: 112, height: 112);

    expect(FaceAlignmentService().align(source, face), isNull);
  });
}
