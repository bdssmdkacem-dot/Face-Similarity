import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../../face_analysis/data/face_detection_service.dart';
import '../../results/presentation/results_page.dart';

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});
  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> {
  CameraController? _controller;
  bool _busy = false;
  String? _message;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _message = 'لم يتم العثور على كاميرا.');
      return;
    }
    final front = cameras.firstWhere((c) => c.lensDirection == CameraLensDirection.front, orElse: () => cameras.first);
    final controller = CameraController(front, ResolutionPreset.medium, enableAudio: false);
    await controller.initialize();
    if (!mounted) return;
    setState(() => _controller = controller);
  }

  Future<void> _scan() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _busy) return;
    setState(() { _busy = true; _message = 'جاري تحليل الوجه…'; });
    try {
      final file = await controller.takePicture();
      final result = await FaceDetectionService().validate(file.path);
      if (!mounted) return;
      if (!result.isValid) {
        setState(() { _busy = false; _message = result.message; });
        return;
      }
      Navigator.of(context).push(MaterialPageRoute(builder: (_) => ResultsPage(imagePath: file.path)));
    } catch (_) {
      if (mounted) setState(() { _busy = false; _message = 'تعذر تحليل الصورة. حاول مرة أخرى.'; });
    }
  }

  @override
  void dispose() { _controller?.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(title: const Text('مسح الوجه')),
      body: controller == null
          ? Center(child: _message == null ? const CircularProgressIndicator() : Text(_message!))
          : Column(children: [
              Expanded(child: CameraPreview(controller)),
              if (_message != null) Padding(padding: const EdgeInsets.all(12), child: Text(_message!)),
              Padding(padding: const EdgeInsets.all(20), child: FilledButton.icon(onPressed: _busy ? null : _scan, icon: const Icon(Icons.auto_awesome), label: const Text('اكتشف شَبَهِي'))),
            ]),
    );
  }
}
