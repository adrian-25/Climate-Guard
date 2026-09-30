# Deployment guide

1. Create a Render web service from this repository; `render.yaml` detects the Docker setup.
2. Attach a persistent disk at `/app/runtime`; SQLite is otherwise lost on redeploy.
3. Set `CORS_ALLOW_ORIGINS` and `PUBLIC_BASE_URL` to the final HTTPS domain.
4. Keep `ALERT_DRY_RUN=true` until a verified sender, native-reviewed translations, and operational ownership are in place.
5. Set `DATABASE_PATH` and `ALERT_DATABASE_PATH` to `/app/runtime/climateguard.sqlite3`.
6. Validate `/health`, `/docs`, dashboard live mode, and unsubscribe links after deployment.

Free tiers can cold-start and may use ephemeral filesystems, so they are unsuitable for durable subscription data without persistent storage.
