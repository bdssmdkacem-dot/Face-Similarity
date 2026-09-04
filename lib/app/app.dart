import 'package:flutter/material.dart';
import '../features/home/presentation/home_page.dart';

class FaceSimilarityApp extends StatelessWidget {
  const FaceSimilarityApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shabah',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorSchemeSeed: const Color(0xFF7C5CFC),
      ),
      home: const HomePage(),
    );
  }
}
