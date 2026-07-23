# Standard Operating Procedure: Factory Deployment 
**System:** YokeClassifier Machine Vision Visual Inspection
**Classification:** Internal Confidential / Deployment Operations

---

## 1. Bill of Materials (BOM) & Equipment Specifications

The deployment architecture is strictly decoupled from the development environment to ensure factory-floor reliability and modular maintainability.

### 1.1 Compute Subsystem
| Component | Recommended Specification | Acceptable Deviations |
| :--- | :--- | :--- |
| **Compute Unit** | Intel N100 Mini PC (Beelink S12 Pro or equivalent) | Any x86/x64 industrial PC or fanless NUC. GPU is **not** required, as the MobileNetV2 inference architecture operates in milliseconds on standard CPUs. |
| **Display** | 7-inch HDMI Display (Waveshare) | Any HDMI/VGA display. Serves as a deterministic PASS/FAIL indicator for the line operator. |

### 1.2 Vision & Optical Subsystem
| Component | Recommended Specification | Acceptable Deviations |
| :--- | :--- | :--- |
| **Industrial Camera** | ELP 2MP USB **Global Shutter** Camera | 🚨 **NON-NEGOTIABLE**. A rolling shutter camera will introduce "jello" artifacts under mechanical vibration. Because the system utilizes strict geometric contour alignment, any sub-pixel spatial warping will severely degrade ROI extraction accuracy. |
| **Illumination** | Industrial LED Ring Light | Diffused Dome Lighting is also acceptable. The core requirement is uniform, shadow-less illumination to ensure the `cv2.Canny` edge detector in the alignment phase extracts accurate boundaries. |

### 1.3 Fixturing & Mounting
| Component | Recommended Specification | Acceptable Deviations |
| :--- | :--- | :--- |
| **Yoke Jig/Fixture** | Custom Milled V-Block | 🚨 **NON-NEGOTIABLE**. The system corrects for microscopic rotation and translation, but the part must be presented to the camera within a ±15° rotational tolerance and bounded spatial constraints. |
| **Z-Axis Mount** | Adjustable Metal Copy Stand | Any rigid aluminum extrusion (e.g., 80/20 framing). Must eliminate Z-axis sway. |

---

## 2. Physical Workcell Assembly

1. **Orthogonal Alignment:** The camera lens must be mounted perfectly perpendicular (orthogonal) to the inspection fixture. Any off-axis tilting will introduce perspective distortion (keystoning).
2. **Concentric Illumination:** Mount the LED ring light concentrically around the camera lens to eliminate directional shadowing on the interior drill holes.
3. **Focal Calibration:** Adjust the Z-axis height until the Yoke occupies approximately 80% to 85% of the total camera FOV. Lock the Z-axis collar securely. *Post-calibration modifications to the Z-axis height will invalidate the configured ROI coordinates.*

---

## 3. System Configuration & Calibration

### 3.1 Software Initialization
1. Provision the Intel N100 with a clean installation of Windows 11 / Linux.
2. Transfer the finalized `inspection_system` repository to `C:\inspection_system`.
3. Install Python 3.12 and dependencies via `pip install -r requirements.txt`.

### 3.2 Master Reference Image Calibration
The YokeClassifier system processes both **Coated** and **Non-Coated** parts. 

However, because the Phase 2 alignment module relies exclusively on the **geometric outer contour** of the part (its physical silhouette) rather than surface texture, the coordinate topology is identical for both finishes. Therefore, the mathematical anchor requires **only one master reference image**.

**Procedure:**
1. Secure a flawless, 100% PERFECT yoke into the physical fixture. You may use *either* a Coated or Non-Coated yoke for this step.
2. Run a standard script to capture a single calibration frame from the ELP camera and save it as `master_reference.jpg`.
3. Move the file to the deployment machine at `C:/inspection_system/perfect/master_reference.jpg`.
4. Update the core routing file (`config/config.yaml`) to establish the new factory optical anchor:
```yaml
paths:
  reference_image: C:/inspection_system/perfect/master_reference.jpg

camera:
  type: webcam      # Switched from ip_camera for production deployment
  index: 0          # Target the ELP USB endpoint
  width: 1920
  height: 1080
```

---

## 4. Production Handover (Kiosk Operation)

The inspection terminal must operate as a headless appliance. Operators should not interact with the operating system or IDEs.

1. **Create the Execution Wrapper:** 
   Generate a batch script (`run_vision_system.bat`):
   ```bat
   @echo off
   cd /d C:\inspection_system
   python phase7_live\live_inspection.py
   ```
2. **Automate Execution on Boot:** 
   Place a shortcut to this batch script inside the OS Startup directory (`shell:startup`). 
3. **Disable Power Management:** 
   Configure Windows Power Settings to `Never Sleep`. The display timeout may be set to 5 minutes to preserve the screen, but the processor must remain active indefinitely.
