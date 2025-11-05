# RMT-Net

RMT-Net is a Django-based project containing two primary apps: `Remote_User` and `Service_Provider`.
It appears to include templates for remote users and service providers, datasets, and an SQLite database for quick local development.

This README provides quick setup and usage instructions so you can run the project locally and explore the codebase.

## Repo layout (important files)

- `manage.py` — Django management entrypoint.
- `rmt_net/` — Django project settings and WSGI/ASGI.
- `Remote_User/`, `Service_Provider/` — main Django apps (models, views, admin, forms).
- `Template/htmls/` — HTML templates used by the app.
- `rmt_net/db.sqlite3` — default local SQLite database (if present).
- `requirements.txt` — Python package dependencies.

## Requirements

- Python 3.8+ (3.8-3.11 recommended).  Use the Python on your system that matches the project environment.
- pip

## Quick local setup

1. Create and activate a virtual environment (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Apply database migrations:

```powershell
python manage.py migrate
```

4. (Optional) Create a superuser for admin access:

```powershell
python manage.py createsuperuser
```

5. Run the development server:

```powershell
python manage.py runserver
```

Then open http://127.0.0.1:8000/ in your browser.

## Database / Data

- By default the project uses SQLite at `rmt_net/db.sqlite3`. If you prefer another DB, update `rmt_net/settings.py` and install the appropriate DB driver.
- There is a `Datasets.csv` file in the repo — review the app code to see how datasets are imported or used.

## Templates and static files

- Templates live in `Template/htmls/` (subfolders for RUser and SProvider). Static assets (images, css) are under `Template/images/` and in the htmls folders.


