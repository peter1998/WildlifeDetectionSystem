// Enhanced API Debugging Tool
console.log("Enhanced API Debug Tool Loaded");

async function testAPI() {
    const results = {
        species: null,
        images: null,
        folders: null,
        imageFile: null,
        annotations: null,
        batchAnnotations: null
    };
    
    try {
        // 1. Test species API
        console.log("Testing species API...");
        const speciesResponse = await fetch('/api/species/');
        results.species = await speciesResponse.json();
        console.log("Species data:", results.species);
        
        // 2. Test images API
        console.log("Testing images API...");
        const imagesResponse = await fetch('/api/images/');
        results.images = await imagesResponse.json();
        console.log("Images data:", results.images);
        
        // 3. Test folders API
        console.log("Testing folders API...");
        const foldersResponse = await fetch('/api/images/folders');
        results.folders = await foldersResponse.json();
        console.log("Folders data:", results.folders);
        
        // If we have images, test image file and annotations
        if (results.images.success && results.images.images && results.images.images.length > 0) {
            const firstImage = results.images.images[0];
            console.log("Found first image:", firstImage);
            
            // 4. Test image file endpoint
            console.log(`Testing image file API for image ID ${firstImage.id}...`);
            const img = new Image();
            img.onload = () => {
                console.log("✅ Image loaded successfully:", img.src);
                results.imageFile = { success: true, width: img.width, height: img.height };
            };
            img.onerror = (e) => {
                console.error("❌ Failed to load image file!", e);
                results.imageFile = { success: false, error: "Failed to load image" };
            };
            img.src = `/api/images/${firstImage.id}/file`;
            
            // 5. Test annotations endpoint
            console.log(`Testing annotations API for image ID ${firstImage.id}...`);
            const annotationsResponse = await fetch(`/api/annotations/image/${firstImage.id}`);
            results.annotations = await annotationsResponse.json();
            console.log("Annotations data:", results.annotations);
            
            // 6. Test batch annotations endpoint (just a dry run)
            const testAnnotation = {
                image_id: firstImage.id,
                annotations: [
                    {
                        species_id: results.species.species[0].id,
                        x_min: 0.1,
                        y_min: 0.1,
                        x_max: 0.2,
                        y_max: 0.2,
                        is_verified: true
                    }
                ]
            };
            
            // Log the payload we would send (but don't actually send it)
            console.log("Example batch annotation payload:", testAnnotation);
        } else {
            console.warn("No images found - can't test image file or annotations endpoints");
        }
        
        return results;
    } catch (error) {
        console.error("API test failed:", error);
        return results;
    }
}

// Run test when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log("Running comprehensive API tests...");
    testAPI().then(results => {
        console.log("API test summary:");
        console.log("- Species API:", results.species ? "✅ Success" : "❌ Failed");
        console.log("- Images API:", results.images ? "✅ Success" : "❌ Failed");
        console.log("- Folders API:", results.folders ? "✅ Success" : "❌ Failed");
        console.log("- Image File API:", results.imageFile ? "✅ Success" : "❌ Failed/Not Tested");
        console.log("- Annotations API:", results.annotations ? "✅ Success" : "❌ Failed/Not Tested");
        
        // Check if we have images
        if (results.images && results.images.success) {
            if (results.images.images.length === 0) {
                console.error("⚠️ No images found in database - index images first");
                alert("No images found in database. Please index images first by visiting /api/images/index-existing");
            } else {
                console.log(`Found ${results.images.images.length} images in database`);
            }
        }
    });
});