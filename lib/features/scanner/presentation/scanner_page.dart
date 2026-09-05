import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../../../core/supabase_config.dart';
import '../../../data/celebrity_match_repository.dart';
import '../../face_analysis/data/face_detection_service.dart';
import '../../face_analysis/data/face_embedding_service.dart';
import '../../results/presentation/results_page.dart';

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage>
    with SingleTickerProviderStateMixin {
  final _faceDetection = FaceDetectionService();
  final _embedding = FaceEmbeddingService();
  CameraController? _controller;
  late final AnimationController _scanAnimation;

  bool _busy = false;
  bool _processingFrame = false;
  bool _autoScanStarted = false;
  bool _aiReady = false;
  int _stableFrames = 0;
  Face? _realtimeFace;
  String _message = 'وجّه وجهك داخل الإطار';
  double _progress = 0;
  String? _aiMessage;

  @override
  void initState() {
    super.initState();
    _scanAnimation = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    )..repeat();
    _initCamera();
    _warmUpAi();
  }

  Future<void> _warmUpAi() async {
    try {
      await _embedding.warmUp(
        onStatus: (status) {
          if (mounted) setState(() => _aiMessage = status);
        },
      );
      if (mounted) {
        setState(() {
          _aiReady = true;
          _aiMessage = '✓ محرك الذكاء الاصطناعي جاهز';
        });
      }
    } catch (error, stack) {
      debugPrint('Shabah AI warm-up failed: $error');
      debugPrintStack(stackTrace: stack);
      if (mounted) {
        setState(() => _aiMessage = 'تعذر تجهيز محرك الذكاء الاصطناعي.');
      }
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) setState(() => _message = 'لم يتم العثور على كاميرا.');
        return;
      }
      final front = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _controller = controller;
        _message = 'وجّه وجهك داخل الإطار';
      });
      await controller.startImageStream(_processCameraImage);
    } catch (error, stack) {
      debugPrint('Shabah camera initialization failed: $error');
      debugPrintStack(stackTrace: stack);
      if (mounted) setState(() => _message = 'تعذر تشغيل الكاميرا.');
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_processingFrame || _busy || _controller == null) return;
    _processingFrame = true;
    try {
      final input = _inputImageFromCameraImage(image);
      if (input == null) return;
      final faces = await _faceDetection.detectRealtime(input);
      if (!mounted || _busy) return;

      if (faces.length != 1) {
        _stableFrames = 0;
        setState(() {
          _realtimeFace = null;
          _progress = 0;
          _message = faces.isEmpty
              ? 'وجّه وجهك داخل الإطار'
              : 'يجب أن يظهر وجه واحد فقط';
        });
        return;
      }

      final face = faces.first;
      final yaw = face.headEulerAngleY?.abs() ?? 0;
      final pitch = face.headEulerAngleX?.abs() ?? 0;
      if (yaw > 25 || pitch > 25) {
        _stableFrames = 0;
        setState(() {
          _realtimeFace = face;
          _progress = 0;
          _message = 'وجّه وجهك نحو الكاميرا مباشرة';
        });
        return;
      }

      _stableFrames++;
      final progress = (_stableFrames / 8).clamp(0.0, 1.0);
      setState(() {
        _realtimeFace = face;
        _progress = progress;
        _message = progress >= 1
            ? '✓ تم تثبيت الوجه — جاري المسح…'
            : 'ثبّت وجهك… ${(progress * 100).round()}%';
      });

      if (_stableFrames >= 8 && !_autoScanStarted && _aiReady) {
        _autoScanStarted = true;
        await _scan();
      }
    } catch (error, stack) {
      debugPrint('Shabah realtime face detection failed: $error');
      debugPrintStack(stackTrace: stack);
    } finally {
      _processingFrame = false;
    }
  }

  InputImage? _inputImageFromCameraImage(CameraImage image) {
    final controller = _controller;
    if (controller == null) return null;

    InputImageRotation? rotation;
    switch (controller.description.sensorOrientation) {
      case 0:
        rotation = InputImageRotation.rotation0deg;
      case 90:
        rotation = InputImageRotation.rotation90deg;
      case 180:
        rotation = InputImageRotation.rotation180deg;
      case 270:
        rotation = InputImageRotation.rotation270deg;
    }
    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null || image.planes.length != 3) return null;

    final bytes = WriteBuffer();
    for (final plane in image.planes) {
      bytes.putUint8List(plane.bytes);
    }

    return InputImage.fromBytes(
      bytes: bytes.done().buffer.asUint8List(),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: image.planes.first.bytesPerRow,
      ),
    );
  }

  Future<void> _deleteCapturedFile(String path) async {
    try {
      final file = File(path);
      if (await file.exists()) await file.delete();
    } catch (_) {}
  }

  Future<void> _scan() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _busy) return;

    setState(() {
      _busy = true;
      _message = 'جاري التقاط أفضل صورة للوجه…';
      _progress = 1;
    });

    String? capturedPath;
    try {
      await controller.stopImageStream();
      final file = await controller.takePicture();
      capturedPath = file.path;

      if (mounted) setState(() => _message = 'جاري التحقق من الوجه…');
      final validation = await _faceDetection.detectAndValidate(file.path);
      if (!validation.isValid || validation.face == null) {
        if (mounted) {
          setState(() {
            _busy = false;
            _autoScanStarted = false;
            _stableFrames = 0;
            _progress = 0;
            _message = validation.message;
          });
          await controller.startImageStream(_processCameraImage);
        }
        return;
      }

      if (!SupabaseConfig.isConfigured) {
        if (mounted) {
          setState(() {
            _busy = false;
            _autoScanStarted = false;
            _message = 'Supabase غير مُهيأ. أضف مفتاح النشر أولاً.';
          });
        }
        return;
      }

      if (mounted) setState(() => _message = '🧠 جاري إنشاء بصمة الوجه…');
      final embedding = await _embedding.embed(
        imagePath: file.path,
        face: validation.face!,
        onStatus: (status) {
          if (mounted) setState(() => _message = status);
        },
      );
      if (!mounted) return;

      setState(() => _message = '🔎 جاري البحث عن أقرب المشاهير…');
      final matches = await CelebrityMatchRepository().matchEmbedding(
        embedding,
        limit: 5,
      );

      if (!mounted) return;
      final resultImagePath = file.path;
      capturedPath = null;
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultsPage(
            matches: matches,
            userImagePath: resultImagePath,
          ),
        ),
      );
      if (mounted) {
        setState(() {
          _busy = false;
          _autoScanStarted = false;
          _stableFrames = 0;
          _progress = 0;
          _realtimeFace = null;
          _message = 'وجّه وجهك داخل الإطار';
        });
        await controller.startImageStream(_processCameraImage);
      }
    } catch (error, stack) {
      debugPrint('Shabah scan failed: $error');
      debugPrintStack(stackTrace: stack);
      if (mounted) {
        setState(() {
          _busy = false;
          _autoScanStarted = false;
          _stableFrames = 0;
          _progress = 0;
          _message = error is TimeoutException
              ? 'استغرق تحليل الوجه وقتًا أطول من المتوقع. حاول مرة أخرى.'
              : 'تعذر إكمال التحليل. تحقق من الاتصال ثم حاول مرة أخرى.';
        });
        try {
          if (!controller.value.isStreamingImages) {
            await controller.startImageStream(_processCameraImage);
          }
        } catch (_) {}
      }
    } finally {
      final path = capturedPath;
      if (path != null) await _deleteCapturedFile(path);
    }
  }

  @override
  void dispose() {
    _scanAnimation.dispose();
    _controller?.dispose();
    _faceDetection.dispose();
    _embedding.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(title: const Text('مسح الوجه')),
      body: controller == null
          ? Center(child: Text(_message))
          : Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(controller),
                _ScanOverlay(
                  progress: _progress,
                  animation: _scanAnimation,
                  hasFace: _realtimeFace != null,
                ),
                Positioned(
                  left: 20,
                  right: 20,
                  bottom: 28,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            _message,
                            textAlign: TextAlign.center,
                            style: const TextStyle(fontWeight: FontWeight.w600),
                          ),
                          if (_aiMessage != null && !_aiReady) ...[
                            const SizedBox(height: 6),
                            Text(
                              _aiMessage!,
                              textAlign: TextAlign.center,
                              style: const TextStyle(fontSize: 12),
                            ),
                          ],
                          const SizedBox(height: 10),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: LinearProgressIndicator(
                              value: _busy ? null : _progress,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _busy
                                ? 'لا تغلق التطبيق أثناء التحليل'
                                : _aiReady
                                    ? 'سيبدأ المسح تلقائيًا عند ثبات الوجه'
                                    : 'جاري تجهيز محرك الذكاء الاصطناعي…',
                            textAlign: TextAlign.center,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}

class _ScanOverlay extends StatelessWidget {
  const _ScanOverlay({
    required this.progress,
    required this.animation,
    required this.hasFace,
  });

  final double progress;
  final Animation<double> animation;
  final bool hasFace;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: AnimatedBuilder(
        animation: animation,
        builder: (context, child) {
          final scanFraction = 0.27 + animation.value * 0.40;
          return CustomPaint(
            painter: _ScanPainter(
              progress: progress,
              scanFraction: scanFraction,
              hasFace: hasFace,
            ),
          );
        },
      ),
    );
  }
}

class _ScanPainter extends CustomPainter {
  const _ScanPainter({
    required this.progress,
    required this.scanFraction,
    required this.hasFace,
  });

  final double progress;
  final double scanFraction;
  final bool hasFace;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.40);
    final rect = Rect.fromCenter(
      center: center,
      width: size.width * 0.70,
      height: size.height * 0.46,
    );
    final radius = Radius.circular(rect.width * 0.32);
    final borderPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..color = hasFace ? Colors.greenAccent : Colors.white70;
    canvas.drawRRect(RRect.fromRectAndRadius(rect, radius), borderPaint);

    final progressPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round
      ..color = Colors.lightBlueAccent;
    final progressRect = rect.deflate(3);
    canvas.drawArc(
      progressRect,
      -math.pi / 2,
      math.pi * 2 * progress,
      false,
      progressPaint,
    );

    if (hasFace) {
      final y = size.height * scanFraction;
      final linePaint = Paint()
        ..strokeWidth = 2
        ..color = Colors.lightBlueAccent.withValues(alpha: 0.9);
      canvas.drawLine(
        Offset(rect.left + 18, y),
        Offset(rect.right - 18, y),
        linePaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _ScanPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.scanFraction != scanFraction ||
        oldDelegate.hasFace != hasFace;
  }
}
