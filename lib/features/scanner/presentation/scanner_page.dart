import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

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

class _ScannerPageState extends State<ScannerPage> {
  final _faceDetection = FaceDetectionService();
  final _embedding = FaceEmbeddingService();

  CameraController? _controller;
  bool _busy = false;
  String? _message;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) setState(() => _message = 'لم يتم العثور على كاميرا.');
        return;
      }

      final front = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _controller = controller);
    } catch (_) {
      if (mounted) setState(() => _message = 'تعذر تشغيل الكاميرا.');
    }
  }

  Future<void> _scan() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _busy) return;

    setState(() {
      _busy = true;
      _message = 'جاري تحليل الوجه…';
    });

    try {
      final file = await controller.takePicture();
      final validation = await _faceDetection.detectAndValidate(file.path);

      if (!validation.isValid || validation.face == null) {
        if (mounted) {
          setState(() {
            _busy = false;
            _message = validation.message;
          });
        }
        return;
      }

      if (!SupabaseConfig.isConfigured) {
        if (mounted) {
          setState(() {
            _busy = false;
            _message = 'Supabase غير مُهيأ. أضف مفتاح النشر أولاً.';
          });
        }
        return;
      }

      setState(() => _message = 'جاري إنشاء بصمة الوجه…');
      final embedding = await _embedding.embed(
        imagePath: file.path,
        face: validation.face!,
      );

      setState(() => _message = 'جاري البحث عن أقرب المشاهير…');
      final matches = await CelebrityMatchRepository().matchEmbedding(
        embedding,
        limit: 5,
      );

      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultsPage(
            imagePath: file.path,
            matches: matches,
          ),
        ),
      );
      setState(() => _busy = false);
    } catch (_) {
      if (mounted) {
        setState(() {
          _busy = false;
          _message = 'تعذر إكمال التحليل. تحقق من الاتصال ثم حاول مرة أخرى.';
        });
      }
    }
  }

  @override
  void dispose() {
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
          ? Center(
              child: _message == null
                  ? const CircularProgressIndicator()
                  : Text(_message!),
            )
          : Column(
              children: [
                Expanded(child: CameraPreview(controller)),
                if (_message != null)
                  Padding(
                    padding: const EdgeInsets.all(12),
                    child: Text(_message!, textAlign: TextAlign.center),
                  ),
                Padding(
                  padding: const EdgeInsets.all(20),
                  child: FilledButton.icon(
                    onPressed: _busy ? null : _scan,
                    icon: const Icon(Icons.auto_awesome),
                    label: const Text('اكتشف شَبَهِي'),
                  ),
                ),
              ],
            ),
    );
  }
}
