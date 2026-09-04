import 'dart:math' as math;
import 'dart:ui' show Point;

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

/// Canonical ArcFace 112x112 landmark template.
/// Order: subject-left eye, subject-right eye, nose, subject-left mouth,
/// subject-right mouth.
class FaceAlignmentService {
  static const outputSize = 112;

  static const _dst = <Point<double>>[
    Point<double>(38.2946, 51.6963),
    Point<double>(73.5318, 51.5014),
    Point<double>(56.0252, 71.7366),
    Point<double>(41.5493, 92.3655),
    Point<double>(70.7299, 92.2041),
  ];

  /// Aligns a decoded image using the same five-point similarity transform
  /// used by the gallery importer.
  ///
  /// Returns null when one of the five required ML Kit landmarks is missing.
  img.Image? align(img.Image source, Face face) {
    final landmarks = <Point<double>>[];
    const types = [
      FaceLandmarkType.leftEye,
      FaceLandmarkType.rightEye,
      FaceLandmarkType.noseBase,
      FaceLandmarkType.leftMouth,
      FaceLandmarkType.rightMouth,
    ];

    for (final type in types) {
      final landmark = face.landmarks[type];
      if (landmark == null) return null;
      landmarks.add(
        Point<double>(
          landmark.position.x.toDouble(),
          landmark.position.y.toDouble(),
        ),
      );
    }

    final transform = _estimateSimilarity(landmarks, _dst);
    if (transform == null) return null;

    final output = img.Image(width: outputSize, height: outputSize);
    for (var y = 0; y < outputSize; y++) {
      for (var x = 0; x < outputSize; x++) {
        // Invert the forward similarity transform: destination -> source.
        final dx = x.toDouble() - transform.tx;
        final dy = y.toDouble() - transform.ty;
        final sourceX = (transform.cos * dx + transform.sin * dy) /
            transform.scale;
        final sourceY = (-transform.sin * dx + transform.cos * dy) /
            transform.scale;

        final pixel = source.getPixelInterpolate(
          sourceX,
          sourceY,
          interpolation: img.Interpolation.linear,
        );
        output.setPixel(x, y, pixel);
      }
    }
    return output;
  }

  _Similarity? _estimateSimilarity(
    List<Point<double>> src,
    List<Point<double>> dst,
  ) {
    if (src.length != 5 || dst.length != 5) return null;

    var srcCx = 0.0;
    var srcCy = 0.0;
    var dstCx = 0.0;
    var dstCy = 0.0;
    for (var i = 0; i < 5; i++) {
      srcCx += src[i].x;
      srcCy += src[i].y;
      dstCx += dst[i].x;
      dstCy += dst[i].y;
    }
    srcCx /= 5;
    srcCy /= 5;
    dstCx /= 5;
    dstCy /= 5;

    var cross = 0.0;
    var dot = 0.0;
    var variance = 0.0;
    for (var i = 0; i < 5; i++) {
      final sx = src[i].x - srcCx;
      final sy = src[i].y - srcCy;
      final dx = dst[i].x - dstCx;
      final dy = dst[i].y - dstCy;
      cross += sx * dy - sy * dx;
      dot += sx * dx + sy * dy;
      variance += sx * sx + sy * sy;
    }

    if (variance <= 1e-9) return null;

    final angle = math.atan2(cross, dot);
    final c = math.cos(angle);
    final s = math.sin(angle);

    var numerator = 0.0;
    for (var i = 0; i < 5; i++) {
      final sx = src[i].x - srcCx;
      final sy = src[i].y - srcCy;
      final rx = c * sx - s * sy;
      final ry = s * sx + c * sy;
      final dx = dst[i].x - dstCx;
      final dy = dst[i].y - dstCy;
      numerator += rx * dx + ry * dy;
    }

    final scale = numerator / variance;
    if (!scale.isFinite || scale <= 1e-6) return null;

    final tx = dstCx - scale * (c * srcCx - s * srcCy);
    final ty = dstCy - scale * (s * srcCx + c * srcCy);
    return _Similarity(scale: scale, cos: c, sin: s, tx: tx, ty: ty);
  }
}

class _Similarity {
  const _Similarity({
    required this.scale,
    required this.cos,
    required this.sin,
    required this.tx,
    required this.ty,
  });

  final double scale;
  final double cos;
  final double sin;
  final double tx;
  final double ty;
}
