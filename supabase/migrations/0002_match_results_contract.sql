-- Harden the public match contract used by the Flutter client.
-- The mobile app needs the celebrity name and primary image in the same RPC response.
create or replace function public.match_celebrity_faces(
  query_embedding extensions.vector(512),
  match_count integer default 5
)
returns table (
  celebrity_id uuid,
  name text,
  similarity float,
  image_url text
)
language sql
stable
security definer
set search_path = public, extensions
as $$
  with ranked as (
    select
      e.celebrity_id,
      (1 - min(e.embedding <=> query_embedding))::float as similarity
    from public.celebrity_embeddings e
    join public.celebs c on c.id = e.celebrity_id
    where c.is_active = true
      and e.embedding is not null
    group by e.celebrity_id
    order by min(e.embedding <=> query_embedding)
    limit greatest(1, least(coalesce(match_count, 5), 20))
  )
  select
    r.celebrity_id,
    c.name,
    r.similarity,
    primary_image.image_url
  from ranked r
  join public.celebs c on c.id = r.celebrity_id
  left join lateral (
    select ci.image_url
    from public.celebrity_images ci
    where ci.celebrity_id = c.id
    order by ci.is_primary desc, ci.created_at asc
    limit 1
  ) primary_image on true
  order by r.similarity desc, c.popularity_score desc, c.name asc;
$$;

-- The Flutter client uses the publishable key, so keep execution available to
-- anonymous/authenticated app sessions while preventing implicit PUBLIC grants.
revoke execute on function public.match_celebrity_faces(extensions.vector(512), integer) from public;
grant execute on function public.match_celebrity_faces(extensions.vector(512), integer) to anon, authenticated;
