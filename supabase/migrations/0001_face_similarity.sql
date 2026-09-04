-- Face Similarity MVP schema
create extension if not exists vector with schema extensions;

create table if not exists public.categories (
  id bigint generated always as identity primary key,
  slug text unique not null,
  name text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.celebs (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  slug text unique not null,
  country_code text,
  gender text,
  bio text,
  popularity_score numeric not null default 0,
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);

create table if not exists public.celebrity_categories (
  celebrity_id uuid not null references public.celebs(id) on delete cascade,
  category_id bigint not null references public.categories(id) on delete cascade,
  primary key (celebrity_id, category_id)
);

create table if not exists public.celebrity_images (
  id uuid primary key default gen_random_uuid(),
  celebrity_id uuid not null references public.celebs(id) on delete cascade,
  image_url text not null,
  source_name text,
  source_url text,
  license_type text,
  is_primary boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists public.celebrity_embeddings (
  id uuid primary key default gen_random_uuid(),
  celebrity_id uuid not null references public.celebs(id) on delete cascade,
  image_id uuid references public.celebrity_images(id) on delete cascade,
  embedding extensions.vector(512),
  quality_score numeric,
  model_name text not null default 'arcface-512',
  created_at timestamptz not null default now()
);

create index if not exists celebrity_embeddings_hnsw_idx
  on public.celebrity_embeddings using hnsw (embedding extensions.vector_cosine_ops);

create table if not exists public.scan_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  device_id text,
  status text not null default 'started',
  processing_ms integer,
  created_at timestamptz not null default now()
);

create table if not exists public.scan_results (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null references public.scan_sessions(id) on delete cascade,
  celebrity_id uuid not null references public.celebs(id),
  rank integer not null,
  similarity numeric not null,
  created_at timestamptz not null default now()
);

create or replace function public.match_celebrity_faces(
  query_embedding extensions.vector(512),
  match_count integer default 5
)
returns table (celebrity_id uuid, celebrity_name text, similarity float)
language sql stable security definer set search_path = public, extensions
as $$
  select e.celebrity_id, c.name,
         (1 - min(e.embedding <=> query_embedding))::float as similarity
  from public.celebrity_embeddings e
  join public.celebs c on c.id = e.celebrity_id
  where c.is_active = true and e.embedding is not null
  group by e.celebrity_id, c.name
  order by min(e.embedding <=> query_embedding)
  limit greatest(1, least(match_count, 20));
$$;

alter table public.categories enable row level security;
alter table public.celebs enable row level security;
alter table public.celebrity_categories enable row level security;
alter table public.celebrity_images enable row level security;
alter table public.celebrity_embeddings enable row level security;
alter table public.scan_sessions enable row level security;
alter table public.scan_results enable row level security;

create policy "public read active celebrities" on public.celebs for select using (is_active = true);
create policy "public read categories" on public.categories for select using (true);
create policy "public read celebrity categories" on public.celebrity_categories for select using (true);
create policy "public read celebrity images" on public.celebrity_images for select using (true);

create policy "users read own scans" on public.scan_sessions for select using (auth.uid() = user_id);
create policy "users read own results" on public.scan_results for select using (
  exists (select 1 from public.scan_sessions s where s.id = scan_id and s.user_id = auth.uid())
);
