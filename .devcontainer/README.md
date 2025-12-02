# Dev Container Setup

## First-time setup

Copy the appropriate override file for your platform:

**Windows/WSL:**
```powershell
Copy-Item .devcontainer/docker-compose.override.wsl.yml .devcontainer/docker-compose.override.yml
```

**Native Linux:**
```bash
cp .devcontainer/docker-compose.override.linux.yml .devcontainer/docker-compose.override.yml
```

Then open the folder in the dev container as normal.

The `docker-compose.override.yml` file is gitignored so each user can maintain their own platform-specific configuration.
