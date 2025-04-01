from app import db
from app.models.models import Species

class SpeciesService:
    """Service for managing wildlife species."""
    
    @staticmethod
    def get_all_species():
        """Get all species."""
        return Species.query.all()
    
    @staticmethod
    def get_species_by_id(species_id):
        """Get a species by ID."""
        return Species.query.get(species_id)
    
    @staticmethod
    def get_species_by_name(name):
        """Get a species by name."""
        return Species.query.filter_by(name=name).first()
    
    @staticmethod
    def create_species(name, scientific_name=None, description=None):
        """Create a new species."""
        # Check if species already exists
        existing = Species.query.filter_by(name=name).first()
        if existing:
            return existing
        
        species = Species(
            name=name,
            scientific_name=scientific_name,
            description=description
        )
        
        try:
            db.session.add(species)
            db.session.commit()
            return species
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def update_species(species_id, name=None, scientific_name=None, description=None):
        """Update an existing species."""
        species = Species.query.get(species_id)
        
        if not species:
            return None
        
        if name:
            species.name = name
        if scientific_name:
            species.scientific_name = scientific_name
        if description:
            species.description = description
        
        try:
            db.session.commit()
            return species
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def delete_species(species_id):
        """Delete a species."""
        species = Species.query.get(species_id)
        
        if not species:
            return False
        
        try:
            db.session.delete(species)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def initialize_default_species():
        """Initialize the database with default species from the project plan."""
        default_species = [
            # Ungulates
            {
                "name": "Red Deer",
                "scientific_name": "Cervus elaphus",
                "description": "Large deer with complex branched antlers"
            },
            {
                "name": "Male Roe Deer",
                "scientific_name": "Capreolus capreolus",
                "description": "Male roe deer with small antlers"
            },
            {
                "name": "Female Roe Deer",
                "scientific_name": "Capreolus capreolus",
                "description": "Female roe deer without antlers"
            },
            {
                "name": "Fallow Deer",
                "scientific_name": "Dama dama",
                "description": "Medium-sized deer with palmate antlers"
            },
            {
                "name": "Wild Boar",
                "scientific_name": "Sus scrofa",
                "description": "Wild pig with tusks"
            },
            {
                "name": "Chamois",
                "scientific_name": "Rupicapra rupicapra",
                "description": "Mountain goat-antelope species"
            },
            
            # Carnivores
            {
                "name": "Fox",
                "scientific_name": "Vulpes vulpes",
                "description": "Red fox"
            },
            {
                "name": "Wolf",
                "scientific_name": "Canis lupus",
                "description": "Gray wolf"
            },
            {
                "name": "Jackal",
                "scientific_name": "Canis aureus",
                "description": "Golden jackal"
            },
            {
                "name": "Brown Bear",
                "scientific_name": "Ursus arctos",
                "description": "Large brown bear"
            },
            {
                "name": "Badger",
                "scientific_name": "Meles meles",
                "description": "European badger"
            },
            {
                "name": "Weasel",
                "scientific_name": "Mustela nivalis",
                "description": "Small carnivorous mammal"
            },
            {
                "name": "Stoat",
                "scientific_name": "Mustela erminea",
                "description": "Short-tailed weasel"
            },
            {
                "name": "Polecat",
                "scientific_name": "Mustela putorius",
                "description": "European polecat"
            },
            {
                "name": "Marten",
                "scientific_name": "Martes martes",
                "description": "Pine marten"
            },
            {
                "name": "Otter",
                "scientific_name": "Lutra lutra",
                "description": "European otter"
            },
            {
                "name": "Wildcat",
                "scientific_name": "Felis silvestris",
                "description": "European wildcat"
            },
            
            # Lagomorphs
            {
                "name": "Rabbit",
                "scientific_name": "Oryctolagus cuniculus",
                "description": "European rabbit"
            },
            {
                "name": "Hare",
                "scientific_name": "Lepus europaeus",
                "description": "European hare"
            },
            
            # Rodents
            {
                "name": "Squirrel",
                "scientific_name": "Sciurus vulgaris",
                "description": "Red squirrel"
            },
            {
                "name": "Dormouse",
                "scientific_name": "Glis glis",
                "description": "Edible dormouse"
            },
            
            # Insectivores
            {
                "name": "Hedgehog",
                "scientific_name": "Erinaceus europaeus",
                "description": "European hedgehog"
            },
            
            # Reptiles
            {
                "name": "Turtle",
                "scientific_name": "Testudo hermanni",
                "description": "Hermann's tortoise"
            },
            
            # Birds
            {
                "name": "Blackbird",
                "scientific_name": "Turdus merula",
                "description": "Common blackbird"
            },
            {
                "name": "Nightingale",
                "scientific_name": "Luscinia megarhynchos",
                "description": "Common nightingale"
            },
            {
                "name": "Pheasant",
                "scientific_name": "Phasianus colchicus",
                "description": "Common pheasant"
            },
            
            # Other
            {
                "name": "Human",
                "scientific_name": "Homo sapiens",
                "description": "Human"
            },
            {
                "name": "Background",
                "scientific_name": None,
                "description": "No animal/background"
            }
        ]
        
        created_count = 0
        
        for species_data in default_species:
            try:
                SpeciesService.create_species(
                    name=species_data["name"],
                    scientific_name=species_data["scientific_name"],
                    description=species_data["description"]
                )
                created_count += 1
            except Exception as e:
                print(f"Error creating species {species_data['name']}: {str(e)}")
        
        return created_count
