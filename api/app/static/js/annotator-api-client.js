// api/app/static/js/annotator-api-client.js

class AnnotatorApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
    }

    // Fetch all species
    async fetchSpecies() {
        try {
            const response = await fetch(`${this.baseUrl}/api/species/`);
            if (!response.ok) {
                throw new Error(`Failed to fetch species: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching species:', error);
            throw error;
        }
    }

    // Fetch all folders
    async fetchFolders() {
        try {
            const response = await fetch(`${this.baseUrl}/api/images/folders`);
            if (!response.ok) {
                throw new Error(`Failed to fetch folders: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching folders:', error);
            throw error;
        }
    }

    // Fetch images (with pagination)
    async fetchImages(page = 1, perPage = 20, folder = '') {
        try {
            let url = `${this.baseUrl}/api/images/?page=${page}&per_page=${perPage}`;
            if (folder) {
                url += `&folder=${encodeURIComponent(folder)}`;
            }
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to fetch images: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching images:', error);
            throw error;
        }
    }

    // Fetch annotations for a specific image
    async fetchAnnotations(imageId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/annotations/image/${imageId}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch annotations: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error fetching annotations for image ${imageId}:`, error);
            throw error;
        }
    }

    // Save batch annotations for an image
    async saveAnnotations(imageId, annotations) {
        try {
            const response = await fetch(`${this.baseUrl}/api/annotations/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_id: imageId,
                    annotations: annotations
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to save annotations: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`Error saving annotations for image ${imageId}:`, error);
            throw error;
        }
    }

    // Mark image as having no animals
    async markNoAnimals(imageId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/images/${imageId}/no-animals`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`Failed to mark image as having no animals: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`Error marking image ${imageId} as having no animals:`, error);
            throw error;
        }
    }

    // Export annotations
    async exportAnnotations(format = 'coco') {
        try {
            const response = await fetch(`${this.baseUrl}/api/annotations/export?format=${format}`);
            if (!response.ok) {
                throw new Error(`Failed to export annotations: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error exporting annotations in ${format} format:`, error);
            throw error;
        }
    }
}

// Export for use in other scripts
window.AnnotatorApiClient = AnnotatorApiClient;