"""
println "="*50
println "TISSUE DETECTION SETUP"
println "="*50

// Clear everything and setup
clearAllObjects()
setImageType('BRIGHTFIELD_H_E')
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581", "Background" : "255 255 255"}')

println "\n1. Creating full image annotation..."
createFullImageAnnotation(true)
println "✓ Full image annotation created"

println "\n2. Detecting tissue regions (to exclude background)..."
selectAnnotations()  // Select the full image annotation

runPlugin('qupath.imagej.detect.tissue.SimpleTissueDetection2', 
    '{"threshold": 210,' +  // Adjust if needed (lower = more sensitive)
    '"requestedPixelSizeMicrons": 20.0,' +  // Fast detection
    '"minAreaMicrons": 5000000.0,' +  // Minimum tissue area (5mm²) to filter out debris
    '"maxHoleAreaMicrons": 1000000.0,' +  // Fill holes smaller than 1mm²
    '"darkBackground": false,' +  // Set true if your background is dark
    '"smoothImage": true,' +
    '"medianCleanup": true,' +
    '"dilateBoundaries": false,' +
    '"smoothCoordinates": true,' +
    '"excludeOnBoundary": false,' +
    '"singleAnnotation": false}')  // Keep separate tissue regions

// Step 3: Remove the original full image annotation, keep only tissue
def allAnnotations = getAnnotationObjects()
def fullImageAnnotation = allAnnotations.find { 
    annotation -> annotation.getROI().getBoundsWidth() == getCurrentImageData().getServer().getWidth() 
}
if (fullImageAnnotation != null) {
    removeObject(fullImageAnnotation, true)
    println "✓ Removed full image annotation, keeping only tissue regions"
}

// Step 4: Check what we found
def tissueAnnotations = getAnnotationObjects()
println "\n3. Tissue detection results:"
println "   Found ${tissueAnnotations.size()} tissue region(s)"

if (tissueAnnotations.isEmpty()) {
    println "   ⚠️ No tissue detected with current threshold"
    println "   Creating full image annotation as fallback..."
    createFullImageAnnotation(true)
    tissueAnnotations = getAnnotationObjects()
} else {
    // Calculate total tissue area
    def totalArea = tissueAnnotations.sum { it.getROI().getArea() }
    def totalAreaMm2 = totalArea * Math.pow(getCurrentImageData().getServer().getPixelCalibration().getAveragedPixelSizeMicrons() / 1000, 2)
    println "   Total tissue area: ${String.format('%.2f', totalAreaMm2)} mm²"
    
    // Report on each tissue region
    tissueAnnotations.eachWithIndex { tissue, idx ->
        def areaMm2 = tissue.getROI().getArea() * Math.pow(getCurrentImageData().getServer().getPixelCalibration().getAveragedPixelSizeMicrons() / 1000, 2)
        println "   Region ${idx + 1}: ${String.format('%.2f', areaMm2)} mm²"
    }
}

// Step 5: Select all tissue regions for processing
selectAnnotations()  // This selects ALL tissue annotations
def selected = getSelectedObjects()
println "\n4. Ready for cell detection:"
println "   ${selected.size()} tissue region(s) selected"
println "   Cell detection will run ONLY in tissue areas (not background)"

println "="*50
println "TISSUE DETECTION COMPLETE - Proceeding to cell detection..."
println "="*50
println ""

// YOUR EXISTING DETECTION CODE CONTINUES HERE
// The detection will now only run within the tissue regions, not in background

// TEST DETECTION ON YOUR WSI
println "\nTESTING DETECTION..."
println "="*50

// Check that we have a selected annotation
//def selected = getSelectedObject()
//if (selected == null) {
//    println "ERROR: No annotation selected!"
//    selectAnnotations()
//    selected = getSelectedObject()
//}

println "Selected region: " + selected
println "Starting detection..."

// Try with settings optimized for your pixel size (0.4942 µm)
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', 
    '{"detectionImageBrightfield": "Optical density sum",' +
    '"requestedPixelSizeMicrons": 0.5,' +  // Close to your 0.4942
    '"backgroundRadiusMicrons": 8.0,' +
    '"backgroundByReconstruction": true,' +
    '"medianRadiusMicrons": 0.0,' +
    '"sigmaMicrons": 1.5,' +
    '"minAreaMicrons": 10.0,' +
    '"maxAreaMicrons": 400.0,' +
    '"threshold": 0.05,' +  // Start with moderate threshold
    '"maxBackground": 2.0,' +
    '"watershedPostProcess": true,' +
    '"excludeDAB": false,' +
    '"cellExpansionMicrons": 5.0,' +
    '"includeNuclei": true,' +
    '"smoothBoundaries": true,' +
    '"makeMeasurements": true}')

def cells = getDetectionObjects()
println "\nDetected: " + cells.size() + " cells"

if (cells.size() == 0) {
    println "\nNo cells found. Trying more sensitive settings..."
    
    // Try again with lower threshold
    clearDetections()
    runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', 
        '{"detectionImageBrightfield": "Optical density sum",' +
        '"requestedPixelSizeMicrons": 0.5,' +
        '"backgroundRadiusMicrons": 4.0,' +  // Less background removal
        '"backgroundByReconstruction": false,' +
        '"medianRadiusMicrons": 0.0,' +
        '"sigmaMicrons": 1.0,' +
        '"minAreaMicrons": 5.0,' +  // Smaller cells
        '"maxAreaMicrons": 500.0,' +
        '"threshold": 0.01,' +  // Much lower threshold
        '"maxBackground": 5.0,' +
        '"watershedPostProcess": true,' +
        '"excludeDAB": false,' +
        '"cellExpansionMicrons": 3.0,' +
        '"includeNuclei": true,' +
        '"smoothBoundaries": true,' +
        '"makeMeasurements": true}')
    
    cells = getDetectionObjects()
    println "With sensitive settings: " + cells.size() + " cells"
}
"""

# Above is a script that finds tissues and then identifies both nuclei and cell boundaries within a WSI