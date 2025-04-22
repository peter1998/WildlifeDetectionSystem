from wtforms import Form, StringField, TextAreaField
from wtforms.validators import DataRequired, Length

class SpeciesForm(Form):
    """Custom form for Species model to avoid tuple/dict confusion in WTForms"""
    name = StringField('Name', validators=[DataRequired(), Length(max=100)], 
                      description='Common name (e.g., "Red Deer")')
    scientific_name = StringField('Scientific Name', validators=[Length(max=255)], 
                                 description='Latin name (e.g., "Cervus elaphus")')
    description = TextAreaField('Description', 
                              description='Additional information about the species')