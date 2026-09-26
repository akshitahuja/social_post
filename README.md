# LinkedIn Content Studio

A local, human-reviewed content pipeline built with CrewAI, Groq, Serper, FastAPI, and Pillow. It researches a topic, writes a LinkedIn post, renders a 1080×1080 infographic, routes targeted revision feedback, and publishes only after a separate confirmation.

Groq support is installed through CrewAI's `litellm` extra; `uv sync` installs it automatically.

Groq rate limits are retried inside the active LLM call. The adapter honors the provider's suggested wait time, adds a 250 ms boundary buffer, and resumes the same agent step. Configure the retry ceiling with `SOCIAL_POST_RATE_LIMIT_MAX_RETRIES` and `SOCIAL_POST_RATE_LIMIT_MAX_WAIT_SECONDS`.

Schema-valid JSON is enforced locally. If a model response is incomplete, the specialist receives the exact Pydantic validation errors and can correct the full object up to `SOCIAL_POST_STRUCTURED_OUTPUT_RETRIES` times before the workflow fails.

## Setup

Prerequisites: `uv` and API keys for Groq and Serper.

```bash
cp .env.example .env
# Add GROQ_API_KEY and SERPER_API_KEY to .env
uv sync --extra dev
uv run social_post
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). `uv` creates and maintains the project-local `.venv` automatically.

## LinkedIn publishing

Generation and export work without LinkedIn credentials. For live publishing:

1. Create a LinkedIn developer application and enable the **Share on LinkedIn** product.
2. Add `http://127.0.0.1:8000/auth/linkedin/callback` as an authorized redirect URL.
3. Set `LINKEDIN_CLIENT_ID` and `LINKEDIN_CLIENT_SECRET` in `.env`.
4. Restart the app and use **Connect LinkedIn**.

The app requests `openid profile w_member_social`, initializes an image upload, uploads the approved PNG, and then creates a personal-profile post. OAuth tokens are stored in `.data/linkedin_token.json` with owner-only permissions and are never committed.

The API version is configurable through `LINKEDIN_API_VERSION`; update it when LinkedIn sunsets a version.

## Workflow behavior

- Ambiguous ideas pause before research.
- Research sources and claims remain visible in the review panel but are excluded from the post body.
- Content-only revisions preserve the current infographic.
- Image-only revisions preserve the current post.
- Approval does not publish. Publishing always requires its own click.
- A successfully published revision records its post URN so repeated publish requests cannot create a duplicate.

All workflow JSON, CrewAI Flow checkpoints, rendered PNGs, and OAuth data live under `.data/`.

## Tests

```bash
uv run pytest
```

The automated suite uses fake agents and mocked LinkedIn HTTP responses. It never calls Groq, Serper, or LinkedIn and cannot create a real post.

## Manual smoke test

1. Start with an ambiguous topic and verify clarification appears before research.
2. Confirm an interpretation and verify the source links, post, infographic, and alt text.
3. Request a content-only revision and confirm the image path is unchanged.
4. Request an image-only revision and confirm the post text is unchanged.
5. Approve the current revision and verify publishing remains a separate action.
6. Connect a LinkedIn test account, publish, and verify the returned post URN is recorded.
7. Click publish again and verify the existing result is returned without another API call.
