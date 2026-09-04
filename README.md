# Face Similarity / Shabah

Privacy-first Flutter app that finds the closest celebrity lookalikes from a face embedding.

## Architecture
- Flutter + Riverpod
- Camera + Google ML Kit for local face validation
- Supabase PostgreSQL + pgvector for celebrity embeddings and nearest-neighbor search
- Edge Functions will host the embedding/search orchestration; secrets never ship in the APK

## Current MVP
- Arabic-first home screen
- Front-camera capture
- Exactly-one-face validation
- Pose sanity checks
- Supabase-ready schema with 512-dimensional embeddings and HNSW cosine index
- Celebrity results contract prepared for the next implementation phase

## Privacy principles
- User photos are not persisted by the MVP.
- Face embeddings from user scans should be treated as sensitive biometric-derived data and should not be retained unless a future feature explicitly requires it with appropriate notice/consent.
- Celebrity source/license metadata is stored alongside imported images.

## Supabase
Apply `supabase/migrations/0001_face_similarity.sql` to the selected project. Do not put a Supabase service-role key in Flutter.

## Next steps
1. Connect the intended Supabase project.
2. Add the embedding service/model and benchmark it against a controlled test set.
3. Import an initial 100-500 celebrity catalog with verified image licensing.
4. Generate embeddings and populate `celebrity_embeddings`.
5. Replace the placeholder ResultsPage with the real RPC search flow.
6. Add CI, Android build, tests, rate limiting, analytics, and privacy/legal screens.
