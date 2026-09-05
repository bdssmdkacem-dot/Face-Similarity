import 'dart:io';

import 'package:flutter/material.dart';

import '../../../data/celebrity_match_repository.dart';

class ResultsPage extends StatefulWidget {
  const ResultsPage({
    super.key,
    required this.matches,
    required this.userImagePath,
  });

  final List<CelebrityMatch> matches;
  final String userImagePath;

  @override
  State<ResultsPage> createState() => _ResultsPageState();
}

class _ResultsPageState extends State<ResultsPage> {
  @override
  void dispose() {
    _deleteUserImage();
    super.dispose();
  }

  Future<void> _deleteUserImage() async {
    try {
      final file = File(widget.userImagePath);
      if (await file.exists()) await file.delete();
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('نتيجتك')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Text(
            'أقرب المشاهير إليك',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          const Text(
            'مقارنة ترفيهية لملامح الوجه، وليست تعرّفًا على الهوية.',
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 20),
          if (widget.matches.isEmpty)
            const Card(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Text(
                  'لا توجد نتائج متاحة حاليًا.',
                  textAlign: TextAlign.center,
                ),
              ),
            )
          else
            ...widget.matches.asMap().entries.map(
              (entry) => _MatchCard(
                rank: entry.key + 1,
                match: entry.value,
                userImagePath: widget.userImagePath,
              ),
            ),
          const SizedBox(height: 20),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('مسح مرة أخرى'),
          ),
        ],
      ),
    );
  }
}

class _MatchCard extends StatelessWidget {
  const _MatchCard({
    required this.rank,
    required this.match,
    required this.userImagePath,
  });

  final int rank;
  final CelebrityMatch match;
  final String userImagePath;

  @override
  Widget build(BuildContext context) {
    final percent = (match.similarity.clamp(0, 1) * 100).toStringAsFixed(1);

    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            Row(
              children: [
                CircleAvatar(child: Text('$rank')),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    match.name,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                Text(
                  '$percent%',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: _FaceImage(
                    imageProvider: FileImage(File(userImagePath)),
                    label: 'أنت',
                  ),
                ),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 35),
                  child: Icon(Icons.compare_arrows, size: 28),
                ),
                Expanded(
                  child: _FaceImage(
                    imageProvider: match.imageUrl == null
                        ? null
                        : NetworkImage(match.imageUrl!),
                    label: match.name,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              'نسبة التشابه: $percent%',
              style: Theme.of(context).textTheme.titleSmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _FaceImage extends StatelessWidget {
  const _FaceImage({required this.imageProvider, required this.label});

  final ImageProvider<Object>? imageProvider;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(14),
          child: AspectRatio(
            aspectRatio: 1,
            child: imageProvider == null
                ? const ColoredBox(
                    color: Color(0xFFECECEC),
                    child: Center(child: Icon(Icons.person, size: 42)),
                  )
                : Image(
                    image: imageProvider!,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => const ColoredBox(
                      color: Color(0xFFECECEC),
                      child: Center(child: Icon(Icons.broken_image, size: 36)),
                    ),
                  ),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          label,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}
