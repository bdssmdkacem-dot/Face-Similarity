import 'package:flutter/material.dart';

import '../../../data/celebrity_match_repository.dart';

class ResultsPage extends StatelessWidget {
  const ResultsPage({
    super.key,
    required this.matches,
  });

  final List<CelebrityMatch> matches;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('نتيجتك')),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          Text(
            'أقرب المشاهير إليك',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          const Text(
            'نتيجة ترفيهية مبنية على تشابه ملامح الوجه، وليست تعرّفًا على الهوية.',
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          if (matches.isEmpty)
            const Card(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Text(
                  'لا توجد بيانات مشاهير كافية حاليًا. سنعرض النتائج بعد تجهيز قاعدة المشاهير.',
                  textAlign: TextAlign.center,
                ),
              ),
            )
          else
            ...matches.asMap().entries.map(
              (entry) => _MatchCard(rank: entry.key + 1, match: entry.value),
            ),
          const SizedBox(height: 24),
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
  const _MatchCard({required this.rank, required this.match});

  final int rank;
  final CelebrityMatch match;

  @override
  Widget build(BuildContext context) {
    final percent = (match.similarity.clamp(0, 1) * 100).toStringAsFixed(1);
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: CircleAvatar(child: Text('$rank')),
        title: Text(match.name),
        subtitle: Text('نسبة التشابه: $percent%'),
        trailing: const Icon(Icons.auto_awesome),
      ),
    );
  }
}
