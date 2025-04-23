#include <hip/hip_runtime.h> // HIP runtime for GPU programming
#include <SDL2/SDL.h>         // SDL2 for windowing, input, and rendering context
#include <SDL2/SDL_ttf.h>       // SDL2_ttf for text rendering
#include <stdio.h>            // Standard C I/O (printf, fprintf)
#include <stdlib.h>           // Standard C library (exit)
#include <cmath>              // Standard C math functions (sinf, fabsf, etc.)
#include <iostream>           // C++ I/O (std::cout, std::cerr - though less used here)
#include <vector>             // C++ standard vector container
#include <string>             // C++ standard string class
#include <sstream>            // C++ string stream for formatting text
#include <stdexcept>          // C++ standard exceptions (std::runtime_error)

// --- Configuration Constants ---

// Base line widths (final apparent width at 1x sampling)
// These are scaled by the SSAA factor for rendering in the supersampled buffer.
const float BASE_LINE_WIDTH = 2.0f;
const float BASE_AXIS_WIDTH = 1.5f;

// Allowed supersampling factors (controls quality vs. performance)
const std::vector<int> ALLOWED_SSAA_FACTORS = {1, 2, 4, 8, 16, 32}; // 1=Off, 2=2x, 4=4x, etc.

// Colors (RGBA format, 0-255)
const unsigned char BG_COLOR[4]   = { 20,  20,  40, 255 }; // Dark blue background
const unsigned char AXIS_COLOR[4] = { 150, 150, 150, 255 }; // Gray axes
const unsigned char FUNC_COLOR[4] = { 0, 255, 0, 255 };   // Green function plot

// Font & On-Screen Text Configuration
const char* FONT_PATH = "DejaVuSansMono.ttf"; // Path to the TTF font file (CHANGE IF NEEDED)
const int FONT_SIZE = 18;                     // Font size in points
const SDL_Color TEXT_COLOR = {200, 200, 220, 255}; // Text color (RGBA)

// Interaction Configuration
const float ZOOM_FACTOR = 1.15f; // Multiplier for zooming in/out

// --- Global State Variables ---

// Screen and Calculation Dimensions
int screenWidth  = 1920; // Target output width (pixels), detected automatically
int screenHeight = 1080; // Target output height (pixels), detected automatically
int superWidth   = 0;    // Current calculation width (supersampled), updated dynamically
int superHeight  = 0;    // Current calculation height (supersampled), updated dynamically
size_t superBufferSize = 0; // Size in bytes of the supersampled GPU buffer

// Supersampling Control
int currentSupersampleFactor = 2; // Initial SSAA factor (e.g., 2 means 2x2)
int currentFactorIndex = 1;       // Index into ALLOWED_SSAA_FACTORS for the current factor

// Mathematical Viewport (controlled by panning/zooming)
float X_MIN = -10.0f;
float X_MAX = 10.0f;
float Y_MIN = -2.0f;
float Y_MAX = 2.0f;

// Mouse Interaction State
bool isDragging = false; // True if the left mouse button is held down for panning
int lastMouseX = 0;      // Last recorded mouse X during drag
int lastMouseY = 0;      // Last recorded mouse Y during drag

// Graph Preset Management
struct GraphPreset {
    std::string name;        // Display name of the function
    int functionIndex;       // Index used in the GPU kernel to select the function
    float initialXMin;       // Default viewport X min for this preset
    float initialXMax;       // Default viewport X max
    float initialYMin;       // Default viewport Y min
    float initialYMax;       // Default viewport Y max
};
std::vector<GraphPreset> presets; // Vector holding all defined graph presets
int currentPresetIndex = 0;       // Index of the currently active preset

// SDL, TTF, and HIP Resources (managed globally for easier cleanup)
SDL_Window* window = nullptr;                // The main application window
SDL_Renderer* renderer = nullptr;            // SDL renderer for drawing
SDL_Texture* graphTexture = nullptr;         // SDL texture holding the final graph image
TTF_Font* font = nullptr;                    // Loaded TTF font resource
unsigned char* h_pixels = nullptr;           // Host (CPU) pixel buffer (screen size)
unsigned char* d_supersampled_pixels = nullptr; // Device (GPU) buffer for high-res rendering (dynamic size)
unsigned char* d_final_pixels = nullptr;     // Device (GPU) buffer for downscaled result (screen size)

// Kernel Launch Configuration
dim3 blockSize(16, 16); // Threads per block (e.g., 16x16 = 256 threads) - adjust based on GPU
dim3 finalGridSize;     // Grid dimensions for kernels operating on the final (screen size) buffer
dim3 superGridSize;     // Grid dimensions for the kernel operating on the supersampled buffer (dynamic)


// --- Forward Declarations ---
void cleanup();                        // Frees all allocated resources
bool updateSupersamplingResources(); // Re-calculates sizes and reallocates GPU memory for SSAA
void setupPresets();                   // Initializes the `presets` vector
void applyPreset(int index);           // Sets the viewport based on a preset
void zoom(float factor, int mouseX, int mouseY); // Handles zooming logic
void pan(int deltaX, int deltaY);      // Handles panning logic


// --- HIP Error Checking Macro ---
// Checks the status of a HIP API call and exits cleanly on error.
#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s (%d) at %s:%d\n", \
                hipGetErrorString(status), status, __FILE__, __LINE__); \
        cleanup(); /* Attempt cleanup before exiting */ \
        exit(EXIT_FAILURE); \
    } \
}

// --- Cleanup Function ---
// Releases all allocated SDL, TTF, and HIP resources in reverse order of creation.
void cleanup() {
    printf("Cleaning up...\n");

    // Free HIP device memory first
    if (d_final_pixels) hipFree(d_final_pixels);
    if (d_supersampled_pixels) hipFree(d_supersampled_pixels);
    d_final_pixels = nullptr;
    d_supersampled_pixels = nullptr;

    // Free host memory
    delete[] h_pixels;
    h_pixels = nullptr;

    // Cleanup SDL_ttf
    if (font) TTF_CloseFont(font);
    font = nullptr;
    TTF_Quit(); // Shut down SDL_ttf subsystem

    // Cleanup SDL
    if (graphTexture) SDL_DestroyTexture(graphTexture);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    graphTexture = nullptr;
    renderer = nullptr;
    window = nullptr;
    SDL_Quit(); // Shut down SDL subsystems

    printf("Cleanup finished.\n");
}

// --- GPU Utility Functions ---

// Clamps a float value within a specified range. Used for color conversion.
__device__ inline float clamp(float val, float minVal, float maxVal) {
    return fminf(maxVal, fmaxf(minVal, val));
}

// --- GPU Kernels ---

// Selects and computes the mathematical function to plot based on an index.
// Executed on the GPU device.
__device__ float functionToPlot(float x, int functionIndex, float yMin_dev, float yMax_dev) {
    // Note: yMin_dev/yMax_dev are passed for functions needing the current view range (like the iterative one).
    switch (functionIndex) {
        case 0:  return sinf(x);                                      // Basic Sine
        case 1:  return cosf(x);                                      // Basic Cosine
        case 2:  return tanf(x);                                      // Tangent (beware asymptotes)
        case 3:  return x * x * 0.1f;                                 // Parabola
        case 4:  return 1.0f / x;                                     // Hyperbola (beware x=0)
        case 5:  return floorf(x);                                    // Floor function
        case 6:  return sinf(x * x);                                  // Chirp-like
        case 7:  return x * sinf(1.0f / x);                           // Interesting oscillations near 0
        case 8:  return sinf(expf(0.5f * x));                         // Faster oscillations
        case 9:  return fabsf(fmodf(x, 4.0f) - 2.0f) - 1.0f;            // Triangle wave
        case 10: return sinf(x) + 0.3f * sinf(x * 5.0f) + 0.1f * cosf(x * 11.0f); // Combination
        case 11: { // Sine + Noise (Example of using braces for local scope in case)
            unsigned int ix = __float_as_uint(x * 1000.0f); // Pseudo-random hash input
            ix = (ix ^ 61) ^ (ix >> 16);                    // Simple hash operations
            ix = ix + (ix << 3);
            ix = ix ^ (ix >> 4);
            ix = ix * 0x27d4eb2d;
            ix = ix ^ (ix >> 15);
            float random = (float)(ix & 0xFFFF) / 65535.0f; // Map hash to [0, 1]
            return sinf(x) + (random - 0.5f) * 0.4f;        // Add noise to sine wave
        }
        case 12: { // Iterative "Fractal" (Example)
            float z_real = 0.0f;
            float z_imag = 0.0f;
            float c_real = x * 0.02f; // Use x as part of the constant
            float c_imag = 0.65f;
            int iter = 0;
            const int max_iter = 40;
            // Iterate a simple complex function (like Mandelbrot/Julia)
            while (z_real * z_real + z_imag * z_imag < 4.0f && iter < max_iter) {
                float temp_real = z_real * z_real - z_imag * z_imag + c_real;
                z_imag = 2.0f * z_real * z_imag + c_imag;
                z_real = temp_real;
                iter++;
            }
            // Map iteration count to the current Y viewport range for visualization
            return (float)iter / (float)max_iter * (yMax_dev - yMin_dev) + yMin_dev;
        }
        default: return 0.0f; // Default case if index is out of range
    }
}


// Kernel 1: Renders the graph at SUPERSAMPLED resolution.
// Each thread calculates the color for one pixel in the high-resolution buffer.
// Uses basic thresholding (no anti-aliasing within this kernel).
__global__ void graphKernelSSAA(unsigned char* pixels, int targetWidth, int targetHeight,
                            float xMin, float xMax, float yMin, float yMax,
                            float lineWidthSuper, float axisWidthSuper, int functionIndex)
{
    // Calculate global pixel coordinates in the supersampled buffer
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: Exit if thread is outside the buffer dimensions
    if (px >= targetWidth || py >= targetHeight) {
        return;
    }

    // --- Coordinate Transformation (Supersampled Pixel -> Math Coordinates) ---
    float xRange = xMax - xMin;
    float yRange = yMax - yMin;
    // Calculate math coordinate corresponding to the center of this supersampled pixel
    float x = xMin + (px + 0.5f) * xRange / (float)targetWidth;
    float y = yMax - (py + 0.5f) * yRange / (float)targetHeight; // Y is inverted

    // Calculate the size of one supersampled pixel in math units (for distance checks)
    float pixelWidthMath = xRange / (float)targetWidth;
    float pixelHeightMath = yRange / (float)targetHeight;

    // --- Calculate Function Value ---
    float func_y = functionToPlot(x, functionIndex, yMin, yMax);
    bool func_valid = !isnan(func_y) && !isinf(func_y); // Check for invalid results (NaN, Infinity)

    // --- Determine Pixel Color (Hard Edges) ---
    // Calculate distance thresholds based on line/axis widths in math units
    float lineThresholdY = (lineWidthSuper / 2.0f) * pixelHeightMath;
    float axisThresholdX = (axisWidthSuper / 2.0f) * pixelWidthMath;
    float axisThresholdY = (axisWidthSuper / 2.0f) * pixelHeightMath;

    // Start with background color
    unsigned char R = BG_COLOR[0];
    unsigned char G = BG_COLOR[1];
    unsigned char B = BG_COLOR[2];
    unsigned char A = BG_COLOR[3];

    // Check if pixel is close to the function curve (y = f(x))
    if (func_valid && func_y >= yMin && func_y <= yMax && fabsf(y - func_y) <= lineThresholdY) {
        R = FUNC_COLOR[0]; G = FUNC_COLOR[1]; B = FUNC_COLOR[2]; A = FUNC_COLOR[3];
    }
    // Check if pixel is close to the Y-axis (x = 0), avoid overwriting function
    else if (0.0f >= xMin && 0.0f <= xMax && fabsf(x) <= axisThresholdX) {
        R = AXIS_COLOR[0]; G = AXIS_COLOR[1]; B = AXIS_COLOR[2]; A = AXIS_COLOR[3];
    }
    // Check if pixel is close to the X-axis (y = 0), avoid overwriting function
    else if (0.0f >= yMin && 0.0f <= yMax && fabsf(y) <= axisThresholdY) {
        R = AXIS_COLOR[0]; G = AXIS_COLOR[1]; B = AXIS_COLOR[2]; A = AXIS_COLOR[3];
    }

    // --- Write RGBA Color to Supersampled Buffer ---
    size_t pixelIndex = ((size_t)py * targetWidth + px) * 4; // Calculate linear index (RGBA = 4 bytes)
    pixels[pixelIndex + 0] = R;
    pixels[pixelIndex + 1] = G;
    pixels[pixelIndex + 2] = B;
    pixels[pixelIndex + 3] = A;
}


// Kernel 2: Downscales the supersampled image to the final screen resolution.
// Each thread calculates one pixel for the final output buffer by averaging
// the corresponding block of pixels from the input (supersampled) buffer.
__global__ void downscaleKernel(const unsigned char* inputPixels, unsigned char* outputPixels,
                                int inputWidth, int inputHeight,
                                int outputWidth, int outputHeight, int factor)
{
    // Calculate global pixel coordinates in the final output buffer
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: Exit if thread is outside the output dimensions
    if (px >= outputWidth || py >= outputHeight) {
        return;
    }

    // Calculate top-left corner of the corresponding block (e.g., 2x2) in the input buffer
    int startX = px * factor;
    int startY = py * factor;

    // Accumulate colors using floats for averaging precision
    float sumR = 0.0f;
    float sumG = 0.0f;
    float sumB = 0.0f;
    float sumA = 0.0f;
    float numSamples = 0.0f; // Count actual samples averaged (for edge cases)

    // Iterate over the block of pixels in the input buffer
    for (int j = 0; j < factor; ++j) {
        for (int i = 0; i < factor; ++i) {
            int currentX = startX + i;
            int currentY = startY + j;

            // Check if the input pixel is within bounds (mostly needed if dimensions aren't exact multiples)
            if (currentX < inputWidth && currentY < inputHeight) {
                size_t inputIndex = ((size_t)currentY * inputWidth + currentX) * 4; // Calculate input linear index
                sumR += inputPixels[inputIndex + 0]; // Add Red component
                sumG += inputPixels[inputIndex + 1]; // Add Green component
                sumB += inputPixels[inputIndex + 2]; // Add Blue component
                sumA += inputPixels[inputIndex + 3]; // Add Alpha component
                numSamples += 1.0f;                  // Increment sample count
            }
        }
    }

    // Calculate the average color and write to the output buffer
    size_t outputIndex = ((size_t)py * outputWidth + px) * 4; // Calculate output linear index
    if (numSamples > 0) {
         // Divide sums by sample count and clamp to valid 0-255 range
         outputPixels[outputIndex + 0] = (unsigned char)clamp(sumR / numSamples, 0.0f, 255.0f);
         outputPixels[outputIndex + 1] = (unsigned char)clamp(sumG / numSamples, 0.0f, 255.0f);
         outputPixels[outputIndex + 2] = (unsigned char)clamp(sumB / numSamples, 0.0f, 255.0f);
         outputPixels[outputIndex + 3] = (unsigned char)clamp(sumA / numSamples, 0.0f, 255.0f);
    } else {
        // Should not happen with proper setup, but provide a fallback (e.g., black pixel)
        outputPixels[outputIndex + 0] = 0;
        outputPixels[outputIndex + 1] = 0;
        outputPixels[outputIndex + 2] = 0;
        outputPixels[outputIndex + 3] = 255; // Opaque alpha
    }
}


// --- Host Helper Functions ---

// Populates the `presets` vector with graph definitions.
void setupPresets() {
    presets.clear(); // Ensure it's empty before adding
    presets.push_back({"y = sin(x)",            0, -10.0f, 10.0f, -1.5f, 1.5f});
    presets.push_back({"y = cos(x)",            1, -10.0f, 10.0f, -1.5f, 1.5f});
    presets.push_back({"y = tan(x)",            2, -8.0f,  8.0f, -10.0f, 10.0f}); // Wider Y for tan
    presets.push_back({"y = 0.1*x^2",           3, -10.0f, 10.0f, -1.0f, 10.0f});
    presets.push_back({"y = 1/x",               4, -5.0f,  5.0f, -5.0f, 5.0f});
    presets.push_back({"y = floor(x)",          5, -10.0f, 10.0f, -5.0f, 5.0f});
    presets.push_back({"y = sin(x^2)",          6, -8.0f,  8.0f, -1.5f, 1.5f});
    presets.push_back({"y = x*sin(1/x)",        7, -1.0f,  1.0f, -1.0f, 1.0f});  // Zoomed in X for detail near 0
    presets.push_back({"y = sin(exp(0.5x))",    8, -5.0f,  5.0f, -1.5f, 1.5f});
    presets.push_back({"Triangle Wave",         9, -10.0f, 10.0f, -1.5f, 1.5f});
    presets.push_back({"Combination",          10, -20.0f, 20.0f, -2.0f, 2.0f});
    presets.push_back({"Sine + Noise",         11, -20.0f, 20.0f, -2.0f, 2.0f});
}

// Sets the current viewport (X_MIN, X_MAX, etc.) to the defaults for the selected preset.
void applyPreset(int index) {
     if (index >= 0 && index < presets.size()) {
         X_MIN = presets[index].initialXMin;
         X_MAX = presets[index].initialXMax;
         Y_MIN = presets[index].initialYMin;
         Y_MAX = presets[index].initialYMax;
         currentPresetIndex = index; // Update the currently selected preset index
         printf("Switched to graph %d: %s\n", index, presets[index].name.c_str());
     } else {
         fprintf(stderr, "Warning: Attempted to apply invalid preset index %d\n", index);
     }
}

// Updates the viewport coordinates to simulate zooming.
// Zooms towards/away from the specified mouse coordinates (mouseX, mouseY).
void zoom(float factor, int mouseX, int mouseY) {
    // 1. Calculate the mathematical coordinate under the mouse cursor
    float mouseMathX = X_MIN + (mouseX / (float)screenWidth) * (X_MAX - X_MIN);
    float mouseMathY = Y_MAX - (mouseY / (float)screenHeight) * (Y_MAX - Y_MIN); // Y is inverted

    // 2. Get current ranges
    float xRange = (X_MAX - X_MIN);
    float yRange = (Y_MAX - Y_MIN);

    // 3. Calculate new ranges based on the zoom factor
    float newXRange = xRange * factor;
    float newYRange = yRange * factor;

    // 4. Calculate new Min/Max values, keeping the point under the mouse stationary
    X_MIN = mouseMathX - (mouseX / (float)screenWidth) * newXRange;
    X_MAX = X_MIN + newXRange;
    Y_MAX = mouseMathY + (mouseY / (float)screenHeight) * newYRange; // Adjust for inverted Y
    Y_MIN = Y_MAX - newYRange;

    // 5. Add constraints to prevent extreme zoom levels (floating point issues)
    const float minRange = 1e-7f;
    const float maxRange = 1e10f;

    if (X_MAX - X_MIN < minRange) {
        float midX = (X_MIN + X_MAX) / 2.0f;
        X_MIN = midX - minRange / 2.0f;
        X_MAX = midX + minRange / 2.0f;
    }
     if (Y_MAX - Y_MIN < minRange) {
        float midY = (Y_MIN + Y_MAX) / 2.0f;
        Y_MIN = midY - minRange / 2.0f;
        Y_MAX = midY + minRange / 2.0f;
    }
     if (X_MAX - X_MIN > maxRange) {
        float midX = X_MIN + xRange / 2.0f; // Use OLD center for stability when hitting max
        X_MIN = midX - maxRange / 2.0f;
        X_MAX = midX + maxRange / 2.0f;
     }
     if (Y_MAX - Y_MIN > maxRange) {
         float midY = Y_MIN + yRange / 2.0f; // Use OLD center
         Y_MIN = midY - maxRange / 2.0f;
         Y_MAX = midY + maxRange / 2.0f;
     }
}

// Updates the viewport coordinates to simulate panning (dragging the view).
// Moves the view based on the change in mouse position (deltaX, deltaY).
void pan(int deltaX, int deltaY) {
    // Calculate the change in mathematical coordinates corresponding to the pixel delta
    float dx = (float)deltaX * (X_MAX - X_MIN) / (float)screenWidth;
    float dy = (float)deltaY * (Y_MAX - Y_MIN) / (float)screenHeight;

    // Update the viewport limits
    X_MIN -= dx;
    X_MAX -= dx;
    Y_MIN += dy; // Add dy because screen Y is inverted relative to math Y
    Y_MAX += dy;
}

// Updates GPU resources when the SSAA factor changes.
// Recalculates dimensions, buffer size, frees the old supersampled buffer (if necessary),
// allocates a new supersampled buffer, and updates the kernel grid dimensions.
// Returns true on success, false on allocation failure.
bool updateSupersamplingResources() {
    // Ensure the index for the factor is valid
    if (currentFactorIndex < 0 || currentFactorIndex >= ALLOWED_SSAA_FACTORS.size()) {
        fprintf(stderr, "Error: Invalid SSAA factor index: %d\n", currentFactorIndex);
        return false; // Should not happen with the modulo logic, but safety first
    }
    // Get the new SSAA factor from the allowed list
    currentSupersampleFactor = ALLOWED_SSAA_FACTORS[currentFactorIndex];

    // Calculate new supersampled dimensions and required buffer size
    int newSuperWidth = screenWidth * currentSupersampleFactor;
    int newSuperHeight = screenHeight * currentSupersampleFactor;
    size_t newSuperBufferSize = (size_t)newSuperWidth * newSuperHeight * 4 * sizeof(unsigned char); // RGBA

    // --- Reallocate GPU memory only if the required size has changed ---
    if (newSuperBufferSize != superBufferSize) {
        printf("Updating SSAA Factor to %dx (CalcRes: %d x %d)\n",
               currentSupersampleFactor, newSuperWidth, newSuperHeight);

        // Free the existing buffer *before* allocating a new one, if it exists
        if (d_supersampled_pixels != nullptr) {
            printf("Freeing previous supersampled buffer (%.2f MB)...\n", (float)superBufferSize / (1024.0f*1024.0f));
            hipError_t freeStatus = hipFree(d_supersampled_pixels);
             d_supersampled_pixels = nullptr; // Mark as potentially freed even if error occurs
             superBufferSize = 0;           // Reset size
            if (freeStatus != hipSuccess) {
                 fprintf(stderr, "HIP Error during hipFree: %s\n", hipGetErrorString(freeStatus));
                 // Depending on the error, continuing might be unsafe. We choose to fail here.
                 return false;
            }
        }

        // Allocate the new supersampled buffer on the GPU
        printf("Allocating new supersampled buffer (%.2f MB)...\n", (float)newSuperBufferSize / (1024.0f*1024.0f));
        hipError_t mallocStatus = hipMalloc(&d_supersampled_pixels, newSuperBufferSize);
        if (mallocStatus != hipSuccess) {
            fprintf(stderr, "HIP Error during hipMalloc: %s\n", hipGetErrorString(mallocStatus));
            fprintf(stderr, " -----> Failed to allocate %.2f MB for supersampling buffer! Try lower SSAA factor or check GPU memory.\n",
                    (float)newSuperBufferSize / (1024.0f*1024.0f));
             d_supersampled_pixels = nullptr; // Ensure pointer is null on failure
             superBufferSize = 0;
             return false; // Indicate failure
        }

        // Update the stored buffer size and dimensions
        superBufferSize = newSuperBufferSize;
        superWidth = newSuperWidth;
        superHeight = newSuperHeight;
        printf("Buffer reallocated successfully.\n");

    } else {
        // Size hasn't changed (e.g., switching from 2x back to 2x), just report
         printf("SSAA Factor set to %dx (CalcRes: %d x %d) - Buffer size unchanged.\n",
                currentSupersampleFactor, newSuperWidth, newSuperHeight);
         // Update dimensions even if size is same (e.g. screen resize scenario - not handled here but good practice)
         superWidth = newSuperWidth;
         superHeight = newSuperHeight;
    }

    // --- Recalculate grid size for the SSAA rendering kernel ---
    // This needs to cover all pixels in the (potentially new) supersampled buffer.
    superGridSize = dim3((superWidth + blockSize.x - 1) / blockSize.x,
                       (superHeight + blockSize.y - 1) / blockSize.y);

    return true; // Indicate success
}


// --- Main Application Entry Point ---
int main(int argc, char* argv[]) {
    // Use a try-catch block for robust error handling during initialization and runtime.
    try {
        // --- Initialization ---
        setupPresets(); // Define the available graphs

        // Initialize SDL Video and TTF subsystems
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error("SDL_Init failed: " + std::string(SDL_GetError()));
        }
        if (TTF_Init() == -1) {
            throw std::runtime_error("TTF_Init failed: " + std::string(TTF_GetError()));
        }

        // Load the font
        font = TTF_OpenFont(FONT_PATH, FONT_SIZE);
        if (!font) {
            throw std::runtime_error("Failed to load font '" + std::string(FONT_PATH) + "': " + TTF_GetError());
        }
        printf("Loaded font: %s\n", FONT_PATH);

        // Get desktop resolution for fullscreen mode
        SDL_DisplayMode dm;
        if (SDL_GetDesktopDisplayMode(0, &dm) != 0) {
            throw std::runtime_error("GetDesktopDisplayMode failed: " + std::string(SDL_GetError()));
        }
        screenWidth = dm.w;
        screenHeight = dm.h;
        printf("Detected screen resolution: %d x %d\n", screenWidth, screenHeight);

        // Create a fullscreen window
        window = SDL_CreateWindow(
            "HIP Graphing Calculator (Dynamic SSAA)", // Window title
            SDL_WINDOWPOS_CENTERED,                   // Initial x position
            SDL_WINDOWPOS_CENTERED,                   // Initial y position
            screenWidth,                              // Width
            screenHeight,                             // Height
            SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_SHOWN // Flags: Fullscreen, shown
        );
        if (!window) {
            throw std::runtime_error("Window creation failed: " + std::string(SDL_GetError()));
        }

        // Create an accelerated SDL renderer with VSync enabled
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) {
            throw std::runtime_error("Renderer creation failed: " + std::string(SDL_GetError()));
        }
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Set color for potential borders
        SDL_RenderClear(renderer);                      // Clear screen initially

        // Create the SDL texture that will receive the final graph image from the CPU buffer
        graphTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, screenWidth, screenHeight);
        if (!graphTexture) {
            throw std::runtime_error("Texture creation failed: " + std::string(SDL_GetError()));
        }

        // Apply the default graph preset
        applyPreset(0);

        // --- HIP Initialization & Memory Allocation ---
        // Allocate host (CPU) memory buffer for final pixel data transfer
        size_t finalBufferSize = (size_t)screenWidth * screenHeight * 4 * sizeof(unsigned char);
        h_pixels = new unsigned char[finalBufferSize];
        if (!h_pixels) {
            throw std::runtime_error("Failed to allocate host memory buffer");
        }

        // Select GPU device (usually device 0) and check for errors
        HIP_CHECK(hipSetDevice(0));

        // Allocate the fixed-size GPU buffer for the final downscaled image
        HIP_CHECK(hipMalloc(&d_final_pixels, finalBufferSize));
        printf("Allocated %.2f MB (Final Output) on GPU\n", (float)finalBufferSize / (1024.0f*1024.0f));

        // Perform initial allocation for the supersampling buffer based on the starting factor
        if (!updateSupersamplingResources()) {
             throw std::runtime_error("Initial allocation of supersampling GPU resources failed!");
        }
        // Note: updateSupersamplingResources also calculates initial superGridSize

        // Calculate the fixed grid size for the downscale kernel (covers screen pixels)
        finalGridSize = dim3((screenWidth + blockSize.x - 1) / blockSize.x,
                           (screenHeight + blockSize.y - 1) / blockSize.y);

        // --- Timing & On-Screen Text Variables ---
        Uint32 lastFPSTime = SDL_GetTicks(); // Time of last FPS calculation
        Uint32 frameCount = 0;               // Frames since last FPS calculation
        float fps = 0.0f;                    // Calculated frames per second
        std::string currentWindowText;       // Text currently displayed on screen

        // --- Main Application Loop ---
        bool quit = false;          // Flag to control loop exit
        SDL_Event event;            // Structure to hold SDL event data
        bool needsRedraw = true;    // Flag to indicate if the graph needs recalculation/redrawing

        printf("\n--- Controls ---\n");
        printf("Mouse Drag: Pan View\n");
        printf("Mouse Wheel: Zoom View\n");
        printf("Left/Right Arrows: Change Graph Preset\n");
        printf("Up/Down Arrows: Change SSAA Factor (Quality/Performance)\n");
        printf("R Key: Reset View for Current Preset\n");
        printf("Esc Key: Quit\n\n");


        while (!quit) {
            // --- Event Handling ---
            // Process all pending events in the queue
            while (SDL_PollEvent(&event)) {
                 switch (event.type) {
                    // Window close button or OS signal
                    case SDL_QUIT:
                        quit = true;
                        break;

                    // Key press events
                    case SDL_KEYDOWN:
                        switch (event.key.keysym.sym) {
                            case SDLK_ESCAPE: // Exit application
                                quit = true;
                                break;
                            case SDLK_LEFT: // Previous graph preset
                                applyPreset((currentPresetIndex - 1 + presets.size()) % presets.size());
                                needsRedraw = true;
                                break;
                            case SDLK_RIGHT: // Next graph preset
                                applyPreset((currentPresetIndex + 1) % presets.size());
                                needsRedraw = true;
                                break;
                            case SDLK_r: // Reset view to preset default
                                applyPreset(currentPresetIndex);
                                needsRedraw = true;
                                break;
                            case SDLK_PLUS:     // Zoom in (alternative)
                            case SDLK_KP_PLUS:  // Zoom in (keypad)
                            case SDLK_EQUALS:   // Zoom in (= key often shares with +)
                                zoom(1.0f / ZOOM_FACTOR, screenWidth / 2, screenHeight / 2); // Zoom towards center
                                needsRedraw = true;
                                break;
                            case SDLK_MINUS:    // Zoom out (alternative)
                            case SDLK_KP_MINUS: // Zoom out (keypad)
                                zoom(ZOOM_FACTOR, screenWidth / 2, screenHeight / 2); // Zoom out from center
                                needsRedraw = true;
                                break;
                            case SDLK_UP: // Increase SSAA factor
                                currentFactorIndex = (currentFactorIndex + 1) % ALLOWED_SSAA_FACTORS.size(); // Cycle through factors
                                if (!updateSupersamplingResources()) {
                                    quit = true; // Exit if reallocation fails (e.g., out of memory)
                                } else {
                                    needsRedraw = true;
                                }
                                break;
                            case SDLK_DOWN: // Decrease SSAA factor
                                currentFactorIndex = (currentFactorIndex - 1 + ALLOWED_SSAA_FACTORS.size()) % ALLOWED_SSAA_FACTORS.size(); // Cycle (wrap around)
                                if (!updateSupersamplingResources()) {
                                    quit = true; // Exit if reallocation fails
                                } else {
                                    needsRedraw = true;
                                }
                                break;
                        } // End switch(event.key.keysym.sym)
                        break; // End case SDL_KEYDOWN

                    // Mouse button down event
                    case SDL_MOUSEBUTTONDOWN:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            isDragging = true;
                            SDL_GetMouseState(&lastMouseX, &lastMouseY); // Record starting position for drag
                        }
                        break;

                    // Mouse button up event
                    case SDL_MOUSEBUTTONUP:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            isDragging = false;
                        }
                        break;

                    // Mouse motion event
                    case SDL_MOUSEMOTION:
                        if (isDragging) {
                            int currentMouseX = event.motion.x;
                            int currentMouseY = event.motion.y;
                            int deltaX = currentMouseX - lastMouseX;
                            int deltaY = currentMouseY - lastMouseY;
                            // Only pan if the mouse actually moved
                            if (deltaX != 0 || deltaY != 0) {
                                pan(deltaX, deltaY);
                                lastMouseX = currentMouseX; // Update last position for next motion event
                                lastMouseY = currentMouseY;
                                needsRedraw = true;
                            }
                        }
                        break;

                    // Mouse wheel scroll event
                    case SDL_MOUSEWHEEL: { // Use braces for local scope of mx, my
                        int mx, my;
                        SDL_GetMouseState(&mx, &my); // Get current mouse position for zoom centering
                        if (event.wheel.y > 0) { // Scroll up/away from user
                            zoom(1.0f / ZOOM_FACTOR, mx, my); // Zoom in
                        } else if (event.wheel.y < 0) { // Scroll down/towards user
                            zoom(ZOOM_FACTOR, mx, my); // Zoom out
                        }
                        needsRedraw = true;
                        } break; // End case SDL_MOUSEWHEEL

                 } // End switch(event.type)
            } // End while(SDL_PollEvent)


            // --- Update Timing and On-Screen Text ---
            Uint32 currentTime = SDL_GetTicks();
            Uint32 deltaFPSTime = currentTime - lastFPSTime;
            frameCount++;
            bool textNeedsUpdate = false; // Flag to regenerate the text string

            // Update FPS calculation periodically
            if (deltaFPSTime >= 500) { // Update roughly twice per second
                fps = (float)frameCount / (deltaFPSTime / 1000.0f);
                lastFPSTime = currentTime;
                frameCount = 0;
                textNeedsUpdate = true; // FPS value changed
            }

            // Also force text update if a redraw was triggered by other input
            if (needsRedraw) {
                textNeedsUpdate = true;
            }

            // Regenerate the on-screen text string if needed
            if (textNeedsUpdate) {
                std::stringstream ss;
                ss.precision(1); // Set precision for FPS display
                ss << "SSAA: " << currentSupersampleFactor << "x | FPS: " << std::fixed << fps // Display SSAA factor and FPS
                   << " | CalcRes: " << superWidth << "x" << superHeight                      // Display calculation resolution
                   << " | OutRes: " << screenWidth << "x" << screenHeight                     // Display output resolution
                   << " | " << presets[currentPresetIndex].name;                              // Display current graph name
                currentWindowText = ss.str(); // Store the generated string

                // Update the window title bar (can be a shorter version if desired)
                SDL_SetWindowTitle(window, ("HIP Graph (DynSSAA): " + presets[currentPresetIndex].name).c_str());
            }


            // --- Render Graph using HIP Kernels ---
            // Only perform GPU work if a redraw is needed and the supersampling buffer is valid
            if (needsRedraw && d_supersampled_pixels != nullptr) {

                // Calculate dynamic line/axis widths based on current SSAA factor
                float lineWidthSuper = BASE_LINE_WIDTH * currentSupersampleFactor;
                float axisWidthSuper = BASE_AXIS_WIDTH * currentSupersampleFactor;

                // --- Kernel Execution ---
                // 1. Launch Kernel 1: Render graph into the high-resolution (supersampled) buffer
                hipLaunchKernelGGL(graphKernelSSAA, superGridSize, blockSize, 0, 0, // Kernel, grid, block, sharedMem, stream
                                   d_supersampled_pixels, superWidth, superHeight,   // Output buffer, dimensions
                                   X_MIN, X_MAX, Y_MIN, Y_MAX,                        // Viewport
                                   lineWidthSuper, axisWidthSuper,                    // Line widths (scaled)
                                   presets[currentPresetIndex].functionIndex);        // Function to plot
                HIP_CHECK(hipGetLastError()); // Check immediately for launch errors

                // 2. Launch Kernel 2: Downscale the high-res buffer to the final screen-size buffer
                //    (Skip direct copy if SSAA factor is 1)
                if (currentSupersampleFactor > 1) {
                     hipLaunchKernelGGL(downscaleKernel, finalGridSize, blockSize, 0, 0,        // Kernel, grid, block, sharedMem, stream
                                        d_supersampled_pixels, d_final_pixels,               // Input, Output buffers
                                        superWidth, superHeight, screenWidth, screenHeight, // Dimensions
                                        currentSupersampleFactor);                          // Downscale factor
                     HIP_CHECK(hipGetLastError()); // Check immediately for launch errors
                } else {
                     // Optimization: If SSAA factor is 1, the "supersampled" buffer is already screen size.
                     // Just copy directly from the SSAA kernel output to the final output buffer on the device.
                     HIP_CHECK(hipMemcpy(d_final_pixels, d_supersampled_pixels, finalBufferSize, hipMemcpyDeviceToDevice));
                }

                // 3. Wait for all GPU operations (both kernels or kernel+copy) to complete
                HIP_CHECK(hipDeviceSynchronize());

                // --- Data Transfer and Texture Update ---
                // 4. Copy the final, screen-sized image from GPU (d_final_pixels) to CPU (h_pixels)
                HIP_CHECK(hipMemcpy(h_pixels, d_final_pixels, finalBufferSize, hipMemcpyDeviceToHost));

                // 5. Update the SDL texture with the data from the CPU buffer (h_pixels)
                void* texturePixels = nullptr; // Pointer to texture memory
                int pitch = 0;                 // Bytes per row in texture memory
                // Lock the texture to get write access
                if (SDL_LockTexture(graphTexture, NULL, &texturePixels, &pitch) == 0) {
                    // Check if texture pitch matches our buffer's row size (screenWidth * 4 bytes/pixel)
                    if (pitch == screenWidth * 4) {
                        // Fast path: Pitch matches, direct memory copy
                        memcpy(texturePixels, h_pixels, finalBufferSize);
                    } else {
                        // Slow path: Pitch mismatch, copy row by row
                        fprintf(stderr, "Warning: Texture pitch (%d) != expected row size (%d). Copying row by row.\n", pitch, screenWidth * 4);
                        unsigned char* src = h_pixels;
                        unsigned char* dst = (unsigned char*)texturePixels;
                        int bytesPerRow = screenWidth * 4;
                        for (int y = 0; y < screenHeight; ++y) {
                            memcpy(dst, src, bytesPerRow);
                            src += bytesPerRow; // Move to next row in source
                            dst += pitch;       // Move to next row in destination texture (using pitch)
                        }
                    }
                    // Unlock the texture after writing
                    SDL_UnlockTexture(graphTexture);
                } else {
                    // Throw error if texture locking fails
                    throw std::runtime_error("Couldn't lock texture: " + std::string(SDL_GetError()));
                }
            } // End if(needsRedraw && d_supersampled_pixels != nullptr)


            // --- Render Scene using SDL Renderer ---
            // This happens every frame, even if the graph wasn't recalculated,
            // to handle potential text updates or OS redraw requests.

            // 1. Copy the (potentially updated) graph texture to the renderer's back buffer.
            //    This overwrites the entire screen with the graph image.
            SDL_RenderCopy(renderer, graphTexture, NULL, NULL); // Src rect, Dst rect (NULL = whole texture/renderer)

            // 2. Render the text overlay on top of the graph texture.
            if (!currentWindowText.empty() && font) {
                 // Render text to an SDL_Surface first
                 SDL_Surface* textSurface = TTF_RenderText_Blended(font, currentWindowText.c_str(), TEXT_COLOR); // Blended mode gives smooth text
                 if (textSurface) {
                     // Create an SDL_Texture from the surface
                     SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
                     if (textTexture) {
                         // Define position and size for the text texture on screen
                         SDL_Rect textRect = { 10, 10, textSurface->w, textSurface->h }; // 10px padding top-left
                         // Copy the text texture to the renderer
                         SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
                         // Clean up the temporary text texture
                         SDL_DestroyTexture(textTexture);
                     } else {
                         // Log warning if texture creation failed
                         fprintf(stderr, "Warning: Failed to create text texture: %s\n", SDL_GetError());
                     }
                     // Clean up the temporary text surface
                     SDL_FreeSurface(textSurface);
                 } else {
                      // Log warning if surface rendering failed
                      fprintf(stderr, "Warning: Failed to render text surface: %s\n", TTF_GetError());
                 }
            } // End if text should be rendered

            // 3. Present the completed frame to the screen.
            //    (Swaps the renderer's back buffer to the front; waits for VSync if enabled).
            SDL_RenderPresent(renderer);

            // Reset the redraw flag after successfully drawing the frame
            needsRedraw = false;

        } // End while(!quit) Main Loop

    // --- Exception Handling & Cleanup ---
    } catch (const std::exception& e) {
        // Catch standard C++ exceptions
        fprintf(stderr, "Fatal Error: %s\n", e.what());
        cleanup(); // Attempt cleanup
        return 1;    // Indicate error exit
    } catch (...) {
        // Catch any other unknown exceptions
        fprintf(stderr, "Fatal Error: Unknown exception occurred.\n");
        cleanup(); // Attempt cleanup
        return 1;    // Indicate error exit
    }

    // --- Normal Exit Cleanup ---
    cleanup(); // Perform cleanup actions on normal loop termination
    return 0;    // Indicate successful execution
}