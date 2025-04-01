from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from app.models.models import Image, Species, Annotation
from app import db

class UniqueModelView(ModelView):
    """Custom ModelView to ensure unique naming and prevent blueprint conflicts"""
    def __init__(self, model, session, **kwargs):
        # Use a unique name and endpoint to prevent conflicts
        kwargs['name'] = kwargs.get('name', f'{model.__name__}_adminview')
        kwargs['endpoint'] = kwargs.get('endpoint', f'{model.__name__.lower()}_admin_endpoint')
        super().__init__(model, session, **kwargs)

def init_admin(app):
    """Initialize Flask-Admin interface with unique blueprint names"""
    admin = Admin(app, name='Wildlife Detection Admin', template_mode='bootstrap3')
    
    # Add views with explicitly unique names and endpoints
    admin.add_view(UniqueModelView(Species, db.session, 
        name='species_admin_view', 
        endpoint='species_admin_endpoint'))
    admin.add_view(UniqueModelView(Image, db.session, 
        name='images_admin_view', 
        endpoint='images_admin_endpoint'))
    admin.add_view(UniqueModelView(Annotation, db.session, 
        name='annotations_admin_view', 
        endpoint='annotations_admin_endpoint'))
    
    return admin