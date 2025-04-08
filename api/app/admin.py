from flask_admin import Admin, BaseView, expose
from flask_admin.contrib.sqla import ModelView
from flask_admin.contrib.fileadmin import FileAdmin
from flask_admin.actions import action
from flask import flash, redirect, url_for, request, Markup, render_template
from app.models.models import Image, Species, Annotation, EnvironmentalData
from app import db
import os
from sqlalchemy import text, func
from datetime import datetime

class BaseModelView(ModelView):
    """Base model view with common improvements"""
    can_view_details = True
    can_export = True
    export_types = ['csv', 'xlsx', 'json']
    page_size = 50
    
    # Format dates nicely in list view
    column_formatters = {
        'created_at': lambda v, c, m, p: m.created_at.strftime('%Y-%m-%d %H:%M:%S') if m.created_at else '',
        'updated_at': lambda v, c, m, p: m.updated_at.strftime('%Y-%m-%d %H:%M:%S') if m.updated_at else '',
        'timestamp': lambda v, c, m, p: m.timestamp.strftime('%Y-%m-%d %H:%M:%S') if m.timestamp else '',
        'upload_date': lambda v, c, m, p: m.upload_date.strftime('%Y-%m-%d %H:%M:%S') if m.upload_date else '',
    }

class SpeciesView(BaseModelView):
    """Custom view for Species model"""
    column_list = ['id', 'name', 'scientific_name', 'description']
    column_searchable_list = ['name', 'scientific_name', 'description']
    column_filters = ['name', 'scientific_name']
    column_sortable_list = ['id', 'name', 'scientific_name']
    column_default_sort = 'id'
    
    # More readable column names
    column_labels = {
        'scientific_name': 'Scientific Name'
    }

class ImageView(BaseModelView):
    """Custom view for Image model"""
    column_list = ['id', 'filename', 'width', 'height', 'upload_date', 'timestamp', 'location', 'camera_id']
    column_searchable_list = ['filename', 'location', 'camera_id']
    column_filters = ['location', 'camera_id', 'upload_date', 'timestamp']
    column_sortable_list = ['id', 'upload_date', 'width', 'height', 'timestamp']
    column_default_sort = ('id', True)
    
    # Display image thumbnail in the list
    column_formatters = {
        **BaseModelView.column_formatters,
        'filename': lambda v, c, m, p: Markup(
            f'<a href="/api/images/{m.id}/file" target="_blank" title="View full image">'
            f'<img src="/api/images/{m.id}/file" width="100" height="75" style="object-fit: cover;">'
            f'</a> {m.filename}'),
    }
    
    # Add links to actions
    column_formatters.update({
        'id': lambda v, c, m, p: Markup(
            f'{m.id} '
            f'<a href="/advanced-annotator?image_id={m.id}" class="btn btn-xs btn-primary" target="_blank">'
            f'<span class="glyphicon glyphicon-pencil"></span> Annotate</a>')
    })

class AnnotationView(BaseModelView):
    """Custom view for Annotation model"""
    column_list = ['id', 'image_id', 'species_id', 'x_min', 'y_min', 'x_max', 'y_max', 
                  'confidence', 'is_verified', 'created_at', 'updated_at']
    column_searchable_list = ['image_id', 'species_id']
    column_filters = ['image_id', 'species_id', 'is_verified', 'created_at']
    column_sortable_list = ['id', 'image_id', 'species_id', 'confidence', 'created_at', 'updated_at']
    column_default_sort = ('id', True)
    
    # Better display of related information
    column_formatters = {
        **BaseModelView.column_formatters,
        'image_id': lambda v, c, m, p: Markup(
            f'<a href="/admin/image_admin_endpoint/details/?id={m.image_id}">{m.image_id}</a>'),
        'species_id': lambda v, c, m, p: Markup(
            f'<a href="/admin/species_admin_endpoint/details/?id={m.species_id}">'
            f'{getattr(m.species, "name", "Unknown") if m.species else "Unknown"}</a>'),
    }
    
    # Add a "Delete All Annotations" action
    @action('delete_all_annotations', 'Delete ALL Annotations', 
            'Are you ABSOLUTELY SURE you want to delete ALL annotation records? This CANNOT be undone!')
    def action_delete_all_annotations(self, ids):
        try:
            # Count annotations before deleting
            count = Annotation.query.count()
            # Delete all annotations
            Annotation.query.delete()
            db.session.commit()
            flash(f'Successfully deleted all {count} annotation records!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting annotations: {str(e)}', 'error')

class EnvironmentalDataView(BaseModelView):
    """Custom view for EnvironmentalData model if it exists"""
    column_list = ['id', 'image_id', 'temperature', 'moon_phase', 'snow_cover', 
                  'light_condition', 'vegetation_type']
    column_searchable_list = ['image_id', 'light_condition', 'vegetation_type']
    column_filters = ['image_id', 'moon_phase', 'snow_cover', 'light_condition']

class DashboardView(BaseView):
    """View for dashboard link"""
    @expose('/')
    def index(self):
        return redirect('/')

class AnnotationStatsView(BaseView):
    """View for annotation statistics and visualizations"""
    @expose('/')
    def index(self):
        try:
            # Query to count annotations by species
            stats = db.session.query(
                Species.id,
                Species.name.label('species_name'),
                func.count(Annotation.id).label('count')
            ).join(
                Annotation, Species.id == Annotation.species_id
            ).group_by(
                Species.id, Species.name
            ).order_by(
                func.count(Annotation.id).desc()
            ).all()
            
            # Format stats for display
            species_stats = [
                {'id': item.id, 'name': item.species_name, 'count': item.count} 
                for item in stats
            ]
            
            # Calculate total annotations
            total_annotations = Annotation.query.count()
            
            # Get recent annotations
            recent_annotations = db.session.query(
                Annotation, Species.name, Image.filename
            ).join(
                Species, Annotation.species_id == Species.id
            ).join(
                Image, Annotation.image_id == Image.id
            ).order_by(
                Annotation.created_at.desc()
            ).limit(10).all()
            
            return self.render('admin/annotation_stats.html', 
                              species_stats=species_stats,
                              total_annotations=total_annotations,
                              recent_annotations=recent_annotations)
        except Exception as e:
            flash(f'Error loading annotation statistics: {str(e)}', 'error')
            return redirect(url_for('admin.index'))

class SimpleSQLiteBrowserView(BaseView):
    """A simpler SQLite table browser that displays record counts"""
    @expose('/')
    def index(self):
        # Get all table names from the database
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        
        # Get counts for each table
        table_counts = {}
        for table in tables:
            try:
                # Use proper SQLAlchemy text() to avoid SQL injection
                query = text(f"SELECT COUNT(*) FROM {table}")
                result = db.session.execute(query)
                count = result.scalar()
                table_counts[table] = count
            except Exception as e:
                table_counts[table] = f"Error: {str(e)}"
        
        return self.render('admin/sqlite_browser_simple.html', tables=tables, table_counts=table_counts)
    
    @expose('/query', methods=['GET', 'POST'])
    def custom_query(self):
        result = None
        error = None
        query_text = ""
        row_count = 0
        
        if request.method == 'POST':
            query_text = request.form.get('query', '')
            
            # Only allow SELECT queries for safety
            if query_text.strip().upper().startswith('SELECT'):
                try:
                    # Use SQLAlchemy text() function to properly handle the query
                    sql_query = text(query_text)
                    result_proxy = db.session.execute(sql_query)
                    
                    # Get column names
                    columns = result_proxy.keys()
                    
                    # Get rows
                    rows = []
                    for row in result_proxy:
                        rows.append(dict(zip(columns, row)))
                    
                    result = {'columns': columns, 'rows': rows}
                    row_count = len(rows)
                except Exception as e:
                    error = str(e)
            else:
                error = "Only SELECT queries are allowed for security reasons."
        
        return self.render('admin/custom_query_simple.html', 
                          result=result, 
                          error=error, 
                          query=query_text,
                          row_count=row_count)

class DeleteActionsView(BaseView):
    """View for dangerous operations like deleting all records"""
    @expose('/')
    def index(self):
        return self.render('admin/delete_actions.html')
    
    @expose('/delete_all_annotations', methods=['POST'])
    def delete_all_annotations(self):
        if request.method == 'POST':
            try:
                # Delete all annotation records
                count = Annotation.query.count()
                Annotation.query.delete()
                db.session.commit()
                flash(f'Successfully deleted all {count} annotation records!', 'success')
            except Exception as e:
                db.session.rollback()
                flash(f'Error deleting annotations: {str(e)}', 'error')
        
        return redirect(url_for('delete_actions.index'))

def init_admin(app):
    """Initialize Flask-Admin interface with improved views"""
    admin = Admin(app, name='Wildlife Detection Admin', template_mode='bootstrap3')
    
    # Add dashboard link as a separate view
    admin.add_view(DashboardView(name='Dashboard', endpoint='dashboard'))
    
    # Add views with explicitly unique names and endpoints
    admin.add_view(SpeciesView(Species, db.session, 
                              name='Species', 
                              endpoint='species_admin_endpoint',
                              category="Database"))
    
    admin.add_view(ImageView(Image, db.session, 
                            name='Images', 
                            endpoint='image_admin_endpoint',
                            category="Database"))
    
    admin.add_view(AnnotationView(Annotation, db.session, 
                                 name='Annotations', 
                                 endpoint='annotation_admin_endpoint',
                                 category="Database"))
    
    # Add EnvironmentalData if it exists in your models
    try:
        # Check if the EnvironmentalData model exists
        if 'EnvironmentalData' in globals():
            admin.add_view(EnvironmentalDataView(EnvironmentalData, db.session, 
                                               name='Environmental Data', 
                                               endpoint='environmental_data_admin_endpoint',
                                               category="Database"))
    except Exception as e:
        print(f"Could not add EnvironmentalData view: {e}")
    
    # Add annotation statistics view
    admin.add_view(AnnotationStatsView(name='Annotation Stats', 
                                     endpoint='annotation_stats',
                                     category="Reports"))
    
    # Add simpler SQLite browser
    admin.add_view(SimpleSQLiteBrowserView(name='SQLite Browser', 
                                          endpoint='sqlite_browser',
                                          category="Tools"))
    
    # Add delete actions view
    admin.add_view(DeleteActionsView(name='Delete Actions', 
                                   endpoint='delete_actions', 
                                   category="Management"))
    
    # Add file browser for raw images
    try:
        path = app.config.get('UPLOAD_FOLDER')
        if path and os.path.exists(path):
            admin.add_view(FileAdmin(path, '/data/raw_images/', 
                                   name='Raw Images', 
                                   endpoint='files_admin_endpoint',
                                   category="Files"))
    except Exception as e:
        print(f"Could not add FileAdmin view: {e}")
    
    # Create templates needed for SQLite browser and annotation stats
    create_sqlite_browser_templates(app)
    
    return admin

def create_sqlite_browser_templates(app):
    """Create the template files needed for SQLite browser"""
    template_dir = os.path.join(app.root_path, 'templates', 'admin')
    os.makedirs(template_dir, exist_ok=True)
    
    # Template for simplified SQLite browser
    sqlite_browser_simple_html = """
{% extends 'admin/master.html' %}
{% block body %}
<div class="container">
    <h1>SQLite Browser</h1>
    <div class="row">
        <div class="col-md-8">
            <div class="panel panel-primary">
                <div class="panel-heading">
                    <h3 class="panel-title">Database Tables</h3>
                </div>
                <div class="panel-body">
                    <table class="table table-striped table-bordered">
                        <thead>
                            <tr>
                                <th>Table Name</th>
                                <th>Record Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for table in tables %}
                            <tr>
                                <td>{{ table }}</td>
                                <td>{{ table_counts[table] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="panel panel-info">
                <div class="panel-heading">
                    <h3 class="panel-title">Custom SQL Query</h3>
                </div>
                <div class="panel-body">
                    <a href="{{ url_for('.custom_query') }}" class="btn btn-primary">
                        <i class="glyphicon glyphicon-console"></i> SQL Query Tool
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # Template for custom SQL query
    custom_query_simple_html = """
{% extends 'admin/master.html' %}
{% block body %}
<div class="container">
    <div class="row">
        <div class="col-md-12">
            <h1>SQL Query Tool</h1>
            <a href="{{ url_for('.index') }}" class="btn btn-default">
                <i class="glyphicon glyphicon-arrow-left"></i> Back to tables
            </a>
            <hr>
            
            <div class="panel panel-primary">
                <div class="panel-heading">
                    <h3 class="panel-title">Enter SQL Query</h3>
                </div>
                <div class="panel-body">
                    <form method="post">
                        <div class="form-group">
                            <div class="alert alert-info">
                                <strong>Note:</strong> Only SELECT queries are allowed for security reasons.
                            </div>
                            <textarea name="query" class="form-control" rows="4" 
                                placeholder="SELECT * FROM annotation LIMIT 10">{{ query }}</textarea>
                        </div>
                        <button type="submit" class="btn btn-success">
                            <i class="glyphicon glyphicon-play"></i> Execute Query
                        </button>
                    </form>
                </div>
            </div>
            
            {% if error %}
            <div class="alert alert-danger">
                <strong>Error:</strong> {{ error }}
            </div>
            {% endif %}
            
            {% if result %}
            <div class="panel panel-success">
                <div class="panel-heading">
                    <h3 class="panel-title">Query Results ({{ row_count }} rows)</h3>
                </div>
                <div class="panel-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-bordered table-hover">
                            <thead>
                                <tr>
                                    {% for column in result.columns %}
                                    <th>{{ column }}</th>
                                    {% endfor %}
                                </tr>
                            </thead>
                            <tbody>
                                {% for row in result.rows %}
                                <tr>
                                    {% for col in result.columns %}
                                    <td>{{ row[col] }}</td>
                                    {% endfor %}
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # Template for delete actions page
    delete_actions_html = """
{% extends 'admin/master.html' %}
{% block body %}
<div class="container">
    <div class="row">
        <div class="col-md-12">
            <h1>Dangerous Operations</h1>
            <p class="lead">This page contains actions that will permanently delete data. Use with caution!</p>
        </div>
    </div>
    
    <div class="row">
        <div class="col-md-6">
            <div class="panel panel-danger">
                <div class="panel-heading">
                    <h3 class="panel-title">Delete All Annotations</h3>
                </div>
                <div class="panel-body">
                    <p>This will permanently delete <strong>ALL</strong> annotation records from the database. This action cannot be undone.</p>
                    
                    <form method="post" action="{{ url_for('delete_actions.delete_all_annotations') }}" onsubmit="return confirm('Are you ABSOLUTELY SURE you want to delete ALL annotation records? This CANNOT be undone!');">
                        <button type="submit" class="btn btn-danger">
                            <i class="glyphicon glyphicon-trash"></i> Delete All Annotations
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # Template for annotation statistics
    annotation_stats_html = """
{% extends 'admin/master.html' %}
{% block body %}
<div class="container">
    <h1>Annotation Statistics</h1>
    
    <div class="row">
        <div class="col-md-12">
            <div class="alert alert-info">
                <h4>Total Annotations: {{ total_annotations }}</h4>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-md-7">
            <div class="panel panel-primary">
                <div class="panel-heading">
                    <h3 class="panel-title">Annotations by Species</h3>
                </div>
                <div class="panel-body">
                    <div style="height: 400px;">
                        <canvas id="speciesChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-5">
            <div class="panel panel-default">
                <div class="panel-heading">
                    <h3 class="panel-title">Species Breakdown</h3>
                </div>
                <div class="panel-body">
                    <table class="table table-striped table-bordered">
                        <thead>
                            <tr>
                                <th>Species</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for stat in species_stats %}
                            <tr>
                                <td>{{ stat.name }}</td>
                                <td>{{ stat.count }}</td>
                                <td>{{ "%.1f"|format(stat.count / total_annotations * 100) if total_annotations else 0 }}%</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-md-12">
            <div class="panel panel-info">
                <div class="panel-heading">
                    <h3 class="panel-title">Recent Annotations</h3>
                </div>
                <div class="panel-body">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Species</th>
                                <th>Image</th>
                                <th>Created At</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for ann, species_name, filename in recent_annotations %}
                            <tr>
                                <td>{{ ann.id }}</td>
                                <td>{{ species_name }}</td>
                                <td>{{ filename }}</td>
                                <td>{{ ann.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Include Chart.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
<script>
    // Prepare data for chart
    var speciesData = {
        labels: [{% for stat in species_stats %}"{{ stat.name }}",{% endfor %}],
        datasets: [{
            label: 'Annotation Count',
            data: [{% for stat in species_stats %}{{ stat.count }},{% endfor %}],
            backgroundColor: [
                '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
                '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400',
                '#16a085', '#27ae60', '#2980b9', '#8e44ad', '#f1c40f'
            ]
        }]
    };
    
    // Create chart
    window.addEventListener('load', function() {
        var ctx = document.getElementById('speciesChart').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'bar',
            data: speciesData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Annotation Distribution by Species'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Annotations'
                        }
                    }
                }
            }
        });
    });
</script>
{% endblock %}
"""
    
    # Write template files
    with open(os.path.join(template_dir, 'sqlite_browser_simple.html'), 'w') as f:
        f.write(sqlite_browser_simple_html)
    
    with open(os.path.join(template_dir, 'custom_query_simple.html'), 'w') as f:
        f.write(custom_query_simple_html)
    
    with open(os.path.join(template_dir, 'delete_actions.html'), 'w') as f:
        f.write(delete_actions_html)
        
    with open(os.path.join(template_dir, 'annotation_stats.html'), 'w') as f:
        f.write(annotation_stats_html)