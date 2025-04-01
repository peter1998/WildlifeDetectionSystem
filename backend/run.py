from app import create_app, db
from app.models.db_models import Image, Annotation, Model

app = create_app()

# Create tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
