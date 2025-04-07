// Global variables
let canvas, ctx;
let currentImage = null;
let images = [];
let currentImageIndex = 0;
let annotations = [];
let selectedAnnotation = null;
let isDrawing = false;
let startX, startY;
let species = [];
let drawingMode = false;
let currentFilter = 'all'; // New variable to track the current filter

// DOM elements
const imageElement = document.getElementById('current-image');
const imageCounter = document.getElementById('image-counter');
const speciesSelect = document.getElementById('species-select');
const annotationsList = document.getElementById('annotations-list');
const selectedBoxInfo = document.getElementById('selected-box-info');

// Initialize the application
async function init() {
    canvas = document.getElementById('annotation-canvas');
    ctx = canvas.getContext('2d');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load species list
    await loadSpecies();
    
    // Load images
    await loadImages();
    
    // Load first image
    if (images.length > 0) {
        loadImage(0);
    }
}

// Set up event listeners
function setupEventListeners() {
    // Navigation
    document.getElementById('prev-image').addEventListener('click', () => {
        if (currentImageIndex > 0) {
            loadImage(currentImageIndex - 1);
        }
    });
    
    document.getElementById('next-image').addEventListener('click', () => {
        if (currentImageIndex < images.length - 1) {
            loadImage(currentImageIndex + 1);
        }
    });
    
    document.getElementById('search-btn').addEventListener('click', searchImage);
    
    // Canvas interaction
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    
    // Annotation controls
    document.getElementById('create-box').addEventListener('click', toggleDrawingMode);
    document.getElementById('delete-box').addEventListener('click', deleteSelectedAnnotation);
    document.getElementById('save-annotations').addEventListener('click', saveAnnotations);
    
    // Image loaded
    imageElement.addEventListener('load', () => {
        canvas.width = imageElement.width;
        canvas.height = imageElement.height;
        drawAnnotations();
    });
    
    // Add filter dropdown event listener (if it exists in the HTML)
    const filterDropdown = document.getElementById('filter-dropdown');
    if (filterDropdown) {
        filterDropdown.addEventListener('change', async function() {
            currentFilter = this.value;
            await loadImages(); // Reload images with the new filter
            
            // Load first image if available
            if (images.length > 0) {
                loadImage(0);
            }
        });
    }
}

// Load species from API
async function loadSpecies() {
    try {
        const response = await fetch('/api/species/');
        const data = await response.json();
        
        if (data.success) {
            species = data.species;
            
            // Update species dropdown
            speciesSelect.innerHTML = '<option value="">Select Species</option>';
            species.forEach(s => {
                const option = document.createElement('option');
                option.value = s.id;
                option.textContent = s.name;
                speciesSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading species:', error);
    }
}

// Load images from API with filtering
async function loadImages() {
    try {
        // Determine the API endpoint based on filter
        let endpoint = '/api/images/';
        if (currentFilter === 'annotated') {
            endpoint = '/api/images/annotated';
        } else if (currentFilter === 'unannotated') {
            endpoint = '/api/images/unannotated';
        }
        
        const response = await fetch(endpoint);
        const data = await response.json();
        
        if (data.success) {
            images = data.images;
            imageCounter.textContent = `Image ${currentImageIndex + 1} of ${images.length}`;
            
            // Update filter counts if elements exist
            updateFilterCounts();
        }
    } catch (error) {
        console.error('Error loading images:', error);
    }
}

// Update filter count badges
async function updateFilterCounts() {
    try {
        // Get count of all images
        const allResponse = await fetch('/api/images/?per_page=1');
        const allData = await allResponse.json();
        
        // Get count of annotated images
        const annotatedResponse = await fetch('/api/images/annotated?per_page=1');
        const annotatedData = await annotatedResponse.json();
        
        // Get count of unannotated images
        const unannotatedResponse = await fetch('/api/images/unannotated?per_page=1');
        const unannotatedData = await unannotatedResponse.json();
        
        // Update count badges if they exist
        const allCountElement = document.getElementById('all-count');
        const annotatedCountElement = document.getElementById('annotated-count');
        const unannotatedCountElement = document.getElementById('unannotated-count');
        
        if (allCountElement) allCountElement.textContent = allData.total;
        if (annotatedCountElement) annotatedCountElement.textContent = annotatedData.total;
        if (unannotatedCountElement) unannotatedCountElement.textContent = unannotatedData.total;
    } catch (error) {
        console.error('Error updating filter counts:', error);
    }
}

// Load a specific image
async function loadImage(index) {
    if (index < 0 || index >= images.length) return;
    
    currentImageIndex = index;
    currentImage = images[index];
    
    // Update counter
    imageCounter.textContent = `Image ${currentImageIndex + 1} of ${images.length}`;
    
    // Load image
    imageElement.src = `/api/images/${currentImage.id}/file`;
    
    // Update annotation status indicator if it exists
    const statusIndicator = document.getElementById('annotation-status');
    if (statusIndicator) {
        if (currentImage.is_annotated) {
            statusIndicator.textContent = 'Annotated';
            statusIndicator.className = 'status-indicator annotated';
        } else {
            statusIndicator.textContent = 'Not Annotated';
            statusIndicator.className = 'status-indicator not-annotated';
        }
    }
    
    // Load annotations for this image
    await loadAnnotations(currentImage.id);
}

// Load annotations for an image
async function loadAnnotations(imageId) {
    try {
        const response = await fetch(`/api/annotations/image/${imageId}`);
        const data = await response.json();
        
        if (data.success) {
            annotations = data.annotations;
            updateAnnotationsList();
        } else {
            annotations = [];
            updateAnnotationsList();
        }
    } catch (error) {
        console.error('Error loading annotations:', error);
        annotations = [];
        updateAnnotationsList();
    }
}

// Update the annotations list in the sidebar
function updateAnnotationsList() {
    annotationsList.innerHTML = '';
    
    if (annotations.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'No annotations';
        annotationsList.appendChild(li);
    } else {
        annotations.forEach((annotation, index) => {
            const li = document.createElement('li');
            li.textContent = `${annotation.species_name || 'Unknown'} (${index + 1})`;
            li.dataset.index = index;
            
            if (selectedAnnotation === index) {
                li.classList.add('selected');
            }
            
            li.addEventListener('click', () => {
                selectAnnotation(index);
            });
            
            annotationsList.appendChild(li);
        });
    }
}

// Select an annotation
function selectAnnotation(index) {
    selectedAnnotation = index;
    updateAnnotationsList();
    updateSelectedBoxInfo();
    drawAnnotations();
}

// Update the selected box info panel
function updateSelectedBoxInfo() {
    if (selectedAnnotation === null || !annotations[selectedAnnotation]) {
        selectedBoxInfo.innerHTML = '<p>No box selected</p>';
    } else {
        const ann = annotations[selectedAnnotation];
        selectedBoxInfo.innerHTML = `
            <p><strong>Species:</strong> ${ann.species_name || 'Unknown'}</p>
            <p><strong>Confidence:</strong> ${ann.confidence ? (ann.confidence * 100).toFixed(2) + '%' : 'N/A'}</p>
            <p><strong>Verified:</strong> ${ann.is_verified ? 'Yes' : 'No'}</p>
            <p><strong>Coordinates:</strong></p>
            <p>x_min: ${ann.x_min.toFixed(3)}, y_min: ${ann.y_min.toFixed(3)}</p>
            <p>x_max: ${ann.x_max.toFixed(3)}, y_max: ${ann.y_max.toFixed(3)}</p>
        `;
    }
}

// Draw all annotations on the canvas
function drawAnnotations() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw image
    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
    
    // Draw annotations
    annotations.forEach((annotation, index) => {
        const x = annotation.x_min * canvas.width;
        const y = annotation.y_min * canvas.height;
        const width = (annotation.x_max - annotation.x_min) * canvas.width;
        const height = (annotation.y_max - annotation.y_min) * canvas.height;
        
        // Set color based on selection state
        if (index === selectedAnnotation) {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 3;
        } else {
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 2;
        }
        
        // Draw box
        ctx.strokeRect(x, y, width, height);
        
        // Draw label
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x, y - 20, 150, 20);
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(annotation.species_name || 'Unknown', x + 5, y - 5);
    });
    
    // Draw current box if drawing
    if (isDrawing && typeof mouse !== 'undefined') {
        const width = startX - mouse.x;
        const height = startY - mouse.y;
        
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        ctx.strokeRect(mouse.x, mouse.y, width, height);
    }
}

// Handle mouse down event
function handleMouseDown(e) {
    if (!drawingMode) {
        // Check if clicking on an existing annotation
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / canvas.width;
        const y = (e.clientY - rect.top) / canvas.height;
        
        let foundIndex = null;
        
        for (let i = 0; i < annotations.length; i++) {
            const ann = annotations[i];
            if (x >= ann.x_min && x <= ann.x_max && y >= ann.y_min && y <= ann.y_max) {
                foundIndex = i;
                break;
            }
        }
        
        selectAnnotation(foundIndex);
        return;
    }
    
    // Start drawing a new box
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    isDrawing = true;
}

// Handle mouse move event
function handleMouseMove(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Redraw
    drawAnnotations();
    
    // Draw current box
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, mouseX - startX, mouseY - startY);
}

// Handle mouse up event
function handleMouseUp(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    // Ensure we have a minimum box size
    if (Math.abs(endX - startX) < 10 || Math.abs(endY - startY) < 10) {
        isDrawing = false;
        return;
    }
    
    // Calculate normalized coordinates
    const x_min = Math.min(startX, endX) / canvas.width;
    const y_min = Math.min(startY, endY) / canvas.height;
    const x_max = Math.max(startX, endX) / canvas.width;
    const y_max = Math.max(startY, endY) / canvas.height;
    
    // Prompt for species selection
    speciesSelect.focus();
    
    // Add to temporary annotations
    annotations.push({
        image_id: currentImage.id,
        species_id: null,
        species_name: null,
        x_min: x_min,
        y_min: y_min,
        x_max: x_max,
        y_max: y_max,
        confidence: null,
        is_verified: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
    });
    
    // Select the new annotation
    selectAnnotation(annotations.length - 1);
    
    // Reset drawing state
    isDrawing = false;
    drawingMode = false;
    document.getElementById('create-box').textContent = 'Create Box';
    
    // Redraw
    drawAnnotations();
}

// Toggle drawing mode
function toggleDrawingMode() {
    drawingMode = !drawingMode;
    document.getElementById('create-box').textContent = drawingMode ? 'Cancel' : 'Create Box';
}

// Delete selected annotation
function deleteSelectedAnnotation() {
    if (selectedAnnotation === null || !annotations[selectedAnnotation]) return;
    
    const ann = annotations[selectedAnnotation];
    
    // If annotation exists on server, delete it
    if (ann.id) {
        fetch(`/api/annotations/${ann.id}`, {
            method: 'DELETE'
        }).then(response => response.json())
        .then(data => {
            if (data.success) {
                annotations.splice(selectedAnnotation, 1);
                selectedAnnotation = null;
                updateAnnotationsList();
                updateSelectedBoxInfo();
                drawAnnotations();
            }
        }).catch(error => {
            console.error('Error deleting annotation:', error);
        });
    } else {
        // Just remove from local array
        annotations.splice(selectedAnnotation, 1);
        selectedAnnotation = null;
        updateAnnotationsList();
        updateSelectedBoxInfo();
        drawAnnotations();
    }
}

// Save all annotations
async function saveAnnotations() {
    if (!currentImage) return;
    
    // Update selected annotation with current species
    if (selectedAnnotation !== null && annotations[selectedAnnotation]) {
        const speciesId = speciesSelect.value;
        if (speciesId) {
            annotations[selectedAnnotation].species_id = parseInt(speciesId);
            annotations[selectedAnnotation].species_name = 
                species.find(s => s.id === parseInt(speciesId))?.name || 'Unknown';
        }
    }
    
    // Save each annotation
    for (const ann of annotations) {
        // Skip annotations without species
        if (!ann.species_id) continue;
        
        // If annotation exists, update it
        if (ann.id) {
            await fetch(`/api/annotations/${ann.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    species_id: ann.species_id,
                    x_min: ann.x_min,
                    y_min: ann.y_min,
                    x_max: ann.x_max,
                    y_max: ann.y_max,
                    is_verified: true
                })
            });
        } else {
            // Otherwise create new annotation
            await fetch('/api/annotations/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_id: currentImage.id,
                    species_id: ann.species_id,
                    x_min: ann.x_min,
                    y_min: ann.y_min,
                    x_max: ann.x_max,
                    y_max: ann.y_max,
                    is_verified: true
                })
            });
        }
    }
    
    // Update the annotation status of the current image
    currentImage.is_annotated = annotations.length > 0;
    
    // Update annotation status indicator if it exists
    const statusIndicator = document.getElementById('annotation-status');
    if (statusIndicator) {
        if (currentImage.is_annotated) {
            statusIndicator.textContent = 'Annotated';
            statusIndicator.className = 'status-indicator annotated';
        } else {
            statusIndicator.textContent = 'Not Annotated';
            statusIndicator.className = 'status-indicator not-annotated';
        }
    }
    
    // Reload annotations from server
    await loadAnnotations(currentImage.id);
    
    // If we're in a filtered view, we might need to reload the image list
    if (currentFilter !== 'all') {
        await loadImages();
    }
}

// Search for an image by filename
function searchImage() {
    const searchText = document.getElementById('image-search').value.toLowerCase();
    
    if (!searchText) return;
    
    const foundIndex = images.findIndex(img => 
        img.filename.toLowerCase().includes(searchText)
    );
    
    if (foundIndex >= 0) {
        loadImage(foundIndex);
    } else {
        alert('No image found with that name');
    }
}

// Initialize when the page loads
window.addEventListener('load', init);