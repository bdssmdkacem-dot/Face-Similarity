import 'package:flutter/material.dart';

class ResultsPage extends StatelessWidget {
  const ResultsPage({super.key, required this.imagePath});
  final String imagePath;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('نتيجتك')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            const Icon(Icons.auto_awesome, size: 72),
            const SizedBox(height: 20),
            Text('تم التحقق من الوجه', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            const Text('محرك المطابقة سيعيد أفضل 5 نتائج من قاعدة المشاهير عند ربط Supabase.'),
            const SizedBox(height: 24),
            FilledButton(onPressed: () => Navigator.of(context).pop(), child: const Text('مسح مرة أخرى')),
          ]),
        ),
      ),
    );
  }
}
