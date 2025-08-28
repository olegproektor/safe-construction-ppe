/**
 * Safe Construction - PPE Detection Web Interface JavaScript
 * 
 * This script handles:
 * - Webcam access and video streaming
 * - Periodic frame capture and transmission
 * - Real-time detection results display
 * - Settings management
 * - User interface interactions
 * 
 * Author: PPE Detection System
 * Date: 2025-08-26
 */

class PPEDetectionApp {
    constructor() {
        console.log('🏠 Initializing PPE Detection App...');
        
        // DOM Elements
        this.videoElement = document.getElementById('videoElement');
        this.captureCanvas = document.getElementById('captureCanvas');
        this.videoOverlay = document.getElementById('videoOverlay');
        this.processingIndicator = document.getElementById('processingIndicator');
        
        // Control buttons
        this.startCameraBtn = document.getElementById('startCameraBtn');
        this.stopCameraBtn = document.getElementById('stopCameraBtn');
        this.startDetectionBtn = document.getElementById('startDetectionBtn');
        this.stopDetectionBtn = document.getElementById('stopDetectionBtn');
        
        // Debug: Check if critical elements exist
        console.log('🔍 DOM Elements found:');
        console.log('  - videoElement:', !!this.videoElement);
        console.log('  - startCameraBtn:', !!this.startCameraBtn);
        console.log('  - stopCameraBtn:', !!this.stopCameraBtn);
        console.log('  - videoOverlay:', !!this.videoOverlay);
        
        if (!this.startCameraBtn) {
            console.error('❌ CRITICAL: Start Camera button not found! ID: startCameraBtn');
        }
        
        if (!this.videoElement) {
            console.error('❌ CRITICAL: Video element not found! ID: videoElement');
        }
        
        // Status elements
        this.connectionStatus = document.getElementById('connectionStatus');
        this.lastUpdateTime = document.getElementById('lastUpdateTime');
        
        // Statistics elements
        this.totalPersons = document.getElementById('totalPersons');
        this.compliantPersons = document.getElementById('compliantPersons');
        this.violationCount = document.getElementById('violationCount');
        this.complianceRate = document.getElementById('complianceRate');
        
        // Performance elements
        this.processingTime = document.getElementById('processingTime');
        this.fpsCounter = document.getElementById('fpsCounter');
        this.frameCounter = document.getElementById('frameCounter');
        
        // Settings elements
        this.settingsPanel = document.getElementById('settingsPanel');
        this.settingsToggleBtn = document.getElementById('settingsToggleBtn');
        this.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        
        // Detection details
        this.detectionDetails = document.getElementById('detectionDetails');
        this.notificationArea = document.getElementById('notificationArea');
        
        // Application state
        this.stream = null;
        this.isDetectionActive = false;
        this.detectionInterval = null;
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        this.fpsHistory = [];
        
        // Enhanced violation tracking - more reasonable for testing
        this.lastViolationSave = 0;
        this.violationSaveInterval = 10000; // Save at most every 10 seconds (reduced for testing)
        this.currentViolationSession = null;
        this.violationSessionTimeout = 60000; // Group violations within 1 minute
        this.violationFrameCount = 0; // Count frames with violations in current session
        this.minViolationFrames = 15; // Require at least 15 frames with violations (reduced for testing)
        this.minSessionDuration = 3000; // Require at least 3 seconds of violations (reduced for testing)
        this.maxSessionDuration = 300000; // Maximum 5 minutes per session
        this.recentSessionFingerprints = new Set(); // Track recent sessions to prevent duplicates
        
        // Settings
        this.settings = {
            confidenceThreshold: 0.5,
            iouThreshold: 0.4,
            ppeOverlapThreshold: 0.3,
            detectionInterval: 500,
            showAnnotations: true,
            autoSaveViolations: false
        };
        
        console.log('🔧 Settings initialized:', this.settings);
        
        // Initialize the application
        this.init();
    }
    
    /**
     * Initialize the application
     */
    init() {
        console.log('🔧 Initializing app...');
        
        // Wait for DOM to be fully ready
        if (document.readyState !== 'complete') {
            console.log('⏳ DOM not ready, waiting...');
            window.addEventListener('load', () => {
                console.log('✅ DOM fully loaded, initializing...');
                this.actualInit();
            });
        } else {
            console.log('✅ DOM already ready, initializing...');
            this.actualInit();
        }
    }
    
    actualInit() {
        console.log('🚀 Starting actual initialization...');
        
        this.setupEventListeners();
        this.setupSettingsControls();
        this.checkAPIConnection();
        this.updateConnectionStatus('offline');
        
        // Final verification
        this.verifyButtonSetup();
        
        console.log('🏗️ Safe Construction PPE Detection App initialized');
    }
    
    verifyButtonSetup() {
        console.log('🔍 Verifying button setup...');
        
        const btn = document.getElementById('startCameraBtn');
        console.log('Button element:', btn);
        console.log('Button disabled:', btn ? btn.disabled : 'N/A');
        console.log('Button onclick:', btn ? btn.onclick : 'N/A');
        console.log('Button event listeners:', btn ? btn.getEventListeners?.() || 'Not available' : 'N/A');
        
        if (btn) {
            console.log('✅ Start Camera button found and ready');
            // Add a test click handler to verify events work
            btn.addEventListener('click', () => {
                console.log('🚨 EMERGENCY CLICK HANDLER TRIGGERED!');
            });
        } else {
            console.error('❌ Start Camera button NOT found!');
        }
    }
    
    /**
     * Set up event listeners for user interactions
     */
    setupEventListeners() {
        console.log('🎬 Setting up event listeners...');
        
        // Check if elements exist
        if (!this.startCameraBtn) {
            console.error('❌ Start Camera button not found!');
            return;
        }
        
        console.log('✅ Start Camera button found:', this.startCameraBtn);
        console.log('Button ID:', this.startCameraBtn.id);
        console.log('Button class:', this.startCameraBtn.className);
        console.log('Button disabled state:', this.startCameraBtn.disabled);
        
        // Camera controls with extensive debugging
        console.log('🔗 Attaching click event to Start Camera button...');
        this.startCameraBtn.addEventListener('click', (event) => {
            console.log('🚨 START CAMERA BUTTON CLICKED!');
            console.log('Event object:', event);
            console.log('Event target:', event.target);
            console.log('Event currentTarget:', event.currentTarget);
            console.log('Button disabled at click:', event.target.disabled);
            
            // Prevent any default behavior
            event.preventDefault();
            event.stopPropagation();
            
            console.log('🚀 About to call startCamera()...');
            this.startCamera();
        });
        
        console.log('✅ Start Camera click event attached successfully');
        
        this.stopCameraBtn.addEventListener('click', () => {
            console.log('⏹️ Stop Camera button clicked');
            this.stopCamera();
        });
        
        this.startDetectionBtn.addEventListener('click', () => {
            console.log('🔍 Start Detection button clicked');
            this.startDetection();
        });
        
        this.stopDetectionBtn.addEventListener('click', () => {
            console.log('⏹️ Stop Detection button clicked');
            this.stopDetection();
        });
        
        // Settings panel
        this.settingsToggleBtn.addEventListener('click', () => this.toggleSettings());
        this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
        
        // Window events
        window.addEventListener('beforeunload', () => this.cleanup());
        window.addEventListener('resize', () => this.handleResize());
        
        console.log('✅ All event listeners set up successfully');
        
        // Test button programmatically
        setTimeout(() => {
            console.log('🧪 Testing button click programmatically in 2 seconds...');
            if (this.startCameraBtn) {
                console.log('Button still exists:', !!this.startCameraBtn);
                console.log('Button parent:', this.startCameraBtn.parentElement);
                console.log('Button is connected to DOM:', this.startCameraBtn.isConnected);
            }
        }, 2000);
    }
    
    /**
     * Set up settings controls and their event listeners
     */
    setupSettingsControls() {
        // Confidence threshold
        const confidenceSlider = document.getElementById('confidenceThreshold');
        const confidenceValue = document.getElementById('confidenceValue');
        confidenceSlider.addEventListener('input', (e) => {
            this.settings.confidenceThreshold = parseFloat(e.target.value);
            confidenceValue.textContent = e.target.value;
        });
        
        // IoU threshold
        const iouSlider = document.getElementById('iouThreshold');
        const iouValue = document.getElementById('iouValue');
        iouSlider.addEventListener('input', (e) => {
            this.settings.iouThreshold = parseFloat(e.target.value);
            iouValue.textContent = e.target.value;
        });
        
        // PPE overlap threshold
        const ppeOverlapSlider = document.getElementById('ppeOverlapThreshold');
        const ppeOverlapValue = document.getElementById('ppeOverlapValue');
        ppeOverlapSlider.addEventListener('input', (e) => {
            this.settings.ppeOverlapThreshold = parseFloat(e.target.value);
            ppeOverlapValue.textContent = e.target.value;
        });
        
        // Detection interval
        const intervalSlider = document.getElementById('detectionInterval');
        const intervalValue = document.getElementById('intervalValue');
        intervalSlider.addEventListener('input', (e) => {
            this.settings.detectionInterval = parseInt(e.target.value);
            intervalValue.textContent = e.target.value + 'ms';
            
            // Restart detection with new interval if active
            if (this.isDetectionActive) {
                this.stopDetection();
                setTimeout(() => this.startDetection(), 100);
            }
        });
        
        // Checkboxes
        document.getElementById('showAnnotations').addEventListener('change', (e) => {
            this.settings.showAnnotations = e.target.checked;
        });
        
        document.getElementById('autoSaveViolations').addEventListener('change', (e) => {
            this.settings.autoSaveViolations = e.target.checked;
        });
    }
    
    /**
     * Check API connection status
     */
    async checkAPIConnection() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                this.updateConnectionStatus('online');
                this.showNotification('API connection established', 'success');
            } else {
                this.updateConnectionStatus('offline');
                this.showNotification('API connection failed', 'error');
            }
        } catch (error) {
            this.updateConnectionStatus('offline');
            this.showNotification('Cannot connect to API server', 'error');
            console.error('API connection error:', error);
        }
    }
    
    /**
     * Start camera access
     */
    async startCamera() {
        console.log('🚀 ====== START CAMERA CALLED ======');
        console.log('Function context:', this);
        console.log('Video element:', this.videoElement);
        console.log('Stream state:', this.stream);
        
        try {
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.error('❌ getUserMedia not supported');
                throw new Error('Camera access not supported by this browser');
            }

            console.log('📱 Requesting camera access...');
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                },
                audio: false
            });
            
            console.log('✅ Camera stream obtained:', this.stream);
            
            this.videoElement.srcObject = this.stream;
            this.videoElement.style.display = 'block';
            this.videoOverlay.style.display = 'none';
            
            // Update button states
            this.startCameraBtn.disabled = true;
            this.stopCameraBtn.disabled = false;
            this.startDetectionBtn.disabled = false;
            
            this.showNotification('Camera started successfully', 'success');
            console.log('📹 Camera started successfully');
            
        } catch (error) {
            console.error('❌ Camera access error:', error);
            
            let errorMessage = 'Failed to access camera';
            let helpText = '';
            
            // Provide specific error messages and help
            if (error.name === 'NotAllowedError' || error.message.includes('Permission denied')) {
                errorMessage = 'Camera permission denied';
                helpText = 'Please allow camera access: Click the 🔒 icon in address bar → Set Camera to "Allow" → Refresh page';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No camera found';
                helpText = 'Please connect a camera or check if another application is using it';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Camera is busy';
                helpText = 'Please close other applications using the camera and try again';
            } else if (error.name === 'OverconstrainedError') {
                errorMessage = 'Camera constraints not supported';
                helpText = 'Your camera doesn\'t support the requested resolution. Trying lower resolution...';
                
                // Try with lower resolution
                this.tryLowerResolution();
                return;
            } else {
                helpText = 'Check browser and system camera permissions';
            }
            
            this.showNotification(errorMessage + ': ' + error.message, 'error');
            this.showCameraHelp(helpText);
            console.error('🔍 Error details:', {
                name: error.name,
                message: error.message,
                stack: error.stack
            });
        }
        
        console.log('🚀 ====== START CAMERA FINISHED ======');
    }
    
    /**
     * Try camera access with lower resolution as fallback
     */
    async tryLowerResolution() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });
            
            this.videoElement.srcObject = this.stream;
            this.videoElement.style.display = 'block';
            this.videoOverlay.style.display = 'none';
            
            // Update button states
            this.startCameraBtn.disabled = true;
            this.stopCameraBtn.disabled = false;
            this.startDetectionBtn.disabled = false;
            
            this.showNotification('Camera started with lower resolution', 'success');
            console.log('📹 Camera started with fallback resolution');
            
        } catch (error) {
            this.showNotification('Camera access failed even with lower resolution: ' + error.message, 'error');
            this.showCameraHelp('Please check camera permissions and try again');
            console.error('Camera fallback error:', error);
        }
    }

    /**
     * Show detailed camera help in the video overlay
     */
    showCameraHelp(helpText) {
        const overlay = this.videoOverlay;
        const overlayContent = overlay.querySelector('.overlay-content');
        
        overlayContent.innerHTML = `
            <h3>📹 Camera Access Required</h3>
            <div class="camera-help">
                <p><strong>Issue:</strong> Camera permission denied or not available</p>
                <p><strong>Solution:</strong> ${helpText}</p>
                <div class="help-steps">
                    <h4>Step-by-step guide:</h4>
                    <ol>
                        <li>Click the 🔒 (lock) or ⚠️ (warning) icon in your browser's address bar</li>
                        <li>Find "Camera" in the permissions list</li>
                        <li>Change it from "Block" or "Ask" to "Allow"</li>
                        <li>Refresh this page (F5 or Ctrl+R)</li>
                        <li>Click "Start Camera" again</li>
                    </ol>
                </div>
                <div class="system-help">
                    <h4>If the problem persists:</h4>
                    <ul>
                        <li>Check Windows Settings → Privacy & security → Camera</li>
                        <li>Enable "Camera access" and "Let desktop apps access your camera"</li>
                        <li>Close other applications that might be using the camera</li>
                        <li>Try a different browser (Chrome, Firefox, Edge)</li>
                    </ul>
                </div>
                <button onclick="location.reload()" class="btn btn-primary" style="margin-top: 15px;">
                    🔄 Refresh Page
                </button>
            </div>
        `;
        
        overlay.style.display = 'flex';
    }
    
    /**
     * Stop camera access
     */
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.videoElement.srcObject = null;
        this.videoElement.style.display = 'none';
        this.videoOverlay.style.display = 'flex';
        
        // Stop detection if active
        if (this.isDetectionActive) {
            this.stopDetection();
        }
        
        // Update button states
        this.startCameraBtn.disabled = false;
        this.stopCameraBtn.disabled = true;
        this.startDetectionBtn.disabled = true;
        this.stopDetectionBtn.disabled = true;
        
        this.showNotification('Camera stopped', 'info');
        console.log('📹 Camera stopped');
    }
    
    /**
     * Start PPE detection
     */
    startDetection() {
        if (!this.stream) {
            this.showNotification('Please start camera first', 'warning');
            return;
        }
        
        this.isDetectionActive = true;
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        
        // Update button states
        this.startDetectionBtn.disabled = true;
        this.stopDetectionBtn.disabled = false;
        
        // Update status
        this.updateConnectionStatus('processing');
        
        // Start detection loop
        this.detectionInterval = setInterval(() => {
            this.captureAndAnalyzeFrame();
        }, this.settings.detectionInterval);
        
        this.showNotification('PPE detection started', 'success');
        console.log('🔍 Detection started');
    }
    
    /**
     * Stop PPE detection
     */
    stopDetection() {
        this.isDetectionActive = false;
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        
        // Update button states
        this.startDetectionBtn.disabled = false;
        this.stopDetectionBtn.disabled = true;
        
        // Update status
        this.updateConnectionStatus('online');
        this.processingIndicator.style.display = 'none';
        
        this.showNotification('PPE detection stopped', 'info');
        console.log('🔍 Detection stopped');
    }
    
    /**
     * Capture frame from video and send for analysis
     */
    async captureAndAnalyzeFrame() {
        if (!this.isDetectionActive || !this.videoElement.videoWidth) {
            return;
        }
        
        try {
            // Show processing indicator
            this.processingIndicator.style.display = 'flex';
            
            // Capture frame
            const canvas = this.captureCanvas;
            const ctx = canvas.getContext('2d');
            
            canvas.width = this.videoElement.videoWidth;
            canvas.height = this.videoElement.videoHeight;
            ctx.drawImage(this.videoElement, 0, 0);
            
            // Convert to blob
            const blob = await new Promise(resolve => 
                canvas.toBlob(resolve, 'image/jpeg', 0.8)
            );
            
            // Send for analysis
            await this.sendFrameForAnalysis(blob);
            
        } catch (error) {
            console.error('Frame capture error:', error);
            this.showNotification('Frame capture failed: ' + error.message, 'error');
        } finally {
            this.processingIndicator.style.display = 'none';
        }
    }
    
    /**
     * Send captured frame to API for analysis
     */
    async sendFrameForAnalysis(blob) {
        const startTime = Date.now();
        
        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            formData.append('conf', this.settings.confidenceThreshold.toString());
            formData.append('iou', this.settings.iouThreshold.toString());
            formData.append('ppe_overlap', this.settings.ppeOverlapThreshold.toString());
            formData.append('draw', 'true');
            formData.append('return_image', this.settings.showAnnotations.toString());
            
            // Send request
            const response = await fetch('/detect/image', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }
            
            const result = await response.json();
            const processingTime = Date.now() - startTime;
            
            // Update results
            await this.updateDetectionResults(result, processingTime);
            
            // Update frame counter and FPS
            this.updatePerformanceMetrics(processingTime);
            
        } catch (error) {
            console.error('API request error:', error);
            this.showNotification('Detection API error: ' + error.message, 'error');
        }
    }
    
    /**
     * Update detection results display
     */
    async updateDetectionResults(result, processingTime) {
        try {
            // Update timestamp
            this.lastUpdateTime.textContent = new Date().toLocaleTimeString();
            
            // Update statistics
            const violationsSummary = result.violations_summary || {};
            this.totalPersons.textContent = violationsSummary.total_persons || 0;
            this.compliantPersons.textContent = violationsSummary.compliant_persons || 0;
            
            // Calculate total violations
            const violationsCount = violationsSummary.violations_count || {};
            const totalViolations = Object.values(violationsCount).reduce((sum, count) => sum + count, 0);
            this.violationCount.textContent = totalViolations;
            
            // Update compliance rate
            const complianceRate = (violationsSummary.compliance_rate || 0) * 100;
            this.complianceRate.textContent = `${complianceRate.toFixed(1)}%`;
            
            // Update detailed results
            this.updateDetailedResults(result);
            
            // Handle violations with enhanced logic
            if (totalViolations > 0 && this.settings.autoSaveViolations) {
                await this.handleViolationDetected(result);
            } else if (this.settings.autoSaveViolations) {
                // Handle case when no violations are detected
                this.handleNoViolations();
            }
            
            // Update processing time
            this.processingTime.textContent = `${processingTime}ms`;
            
        } catch (error) {
            console.error('Error updating results:', error);
        }
    }
    
    /**
     * Update detailed detection results
     */
    updateDetailedResults(result) {
        const detailsContainer = this.detectionDetails;
        
        if (!result.persons || result.persons.length === 0) {
            detailsContainer.innerHTML = '<p class="no-data">No persons detected in current frame</p>';
            return;
        }
        
        let html = '<div class="detection-summary">';
        
        result.persons.forEach((person, index) => {
            const isCompliant = person.has_helmet && person.has_vest && person.violations.length === 0;
            const statusClass = isCompliant ? 'compliant' : 'violation';
            const statusIcon = isCompliant ? '✅' : '⚠️';
            
            html += `
                <div class="person-status ${statusClass}">
                    <div class="person-header">
                        <span class="status-icon">${statusIcon}</span>
                        <strong>Person ${index + 1}</strong>
                        <span class="compliance-badge ${statusClass}">
                            ${isCompliant ? 'Compliant' : 'Violations'}
                        </span>
                    </div>
                    <div class="person-details">
                        <div class="ppe-status">
                            <span class="ppe-item ${person.has_helmet ? 'present' : 'missing'}">
                                ${person.has_helmet ? '🎩' : '❌'} Helmet: ${person.has_helmet ? 'Present' : 'Missing'}
                            </span>
                            <span class="ppe-item ${person.has_vest ? 'present' : 'missing'}">
                                ${person.has_vest ? '🦺' : '❌'} Vest: ${person.has_vest ? 'Present' : 'Missing'}
                            </span>
                        </div>
                        ${person.violations.length > 0 ? `
                            <div class="violations-list">
                                <strong>Violations:</strong> ${person.violations.join(', ')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        detailsContainer.innerHTML = html;
    }
    
    /**
     * Update performance metrics
     */
    updatePerformanceMetrics(processingTime) {
        this.frameCount++;
        this.frameCounter.textContent = this.frameCount;
        
        // Calculate FPS
        const currentTime = Date.now();
        const timeDiff = currentTime - this.lastFrameTime;
        const currentFPS = 1000 / timeDiff;
        
        this.fpsHistory.push(currentFPS);
        if (this.fpsHistory.length > 10) {
            this.fpsHistory.shift();
        }
        
        const avgFPS = this.fpsHistory.reduce((sum, fps) => sum + fps, 0) / this.fpsHistory.length;
        this.fpsCounter.textContent = avgFPS.toFixed(1);
        
        this.lastFrameTime = currentTime;
    }
    
    /**
     * Handle detected violations with enhanced session management
     */
    async handleViolationDetected(result) {
        const timestamp = new Date().toISOString();
        const currentTime = Date.now();
        
        // Count violation frames
        this.violationFrameCount++;
        
        // Start or update violation session
        if (!this.currentViolationSession) {
            this.currentViolationSession = {
                start: timestamp,
                startTime: currentTime,
                totalViolations: 0,
                violationTypes: new Set()
            };
            console.log('🔥 Started new violation session');
        }
        
        // Track violation types in this session
        const violationsCount = result.violations_summary?.violations_count || {};
        Object.keys(violationsCount).forEach(type => {
            if (violationsCount[type] > 0) {
                this.currentViolationSession.violationTypes.add(type);
            }
        });
        
        // Check if we should save this session
        const shouldSave = this.shouldSaveViolationSession(currentTime, result);
        
        if (shouldSave) {
            await this.saveViolationSession(currentTime, result);
        }
    }
    
    /**
     * Enhanced logic to determine if we should save a violation session
     */
    shouldSaveViolationSession(currentTime, result) {
        // Must have an active session
        if (!this.currentViolationSession) return false;
        
        // Check global rate limiting
        const timeSinceLastSave = currentTime - this.lastViolationSave;
        if (timeSinceLastSave < this.violationSaveInterval) {
            return false;
        }
        
        // Check session duration requirements
        const sessionDuration = currentTime - this.currentViolationSession.startTime;
        if (sessionDuration < this.minSessionDuration) {
            return false;
        }
        
        // Check minimum violation frames
        if (this.violationFrameCount < this.minViolationFrames) {
            return false;
        }
        
        // Check maximum session duration (force save if too long)
        if (sessionDuration > this.maxSessionDuration) {
            console.log('⏰ Session too long, forcing save');
            return true;
        }
        
        // Additional validation: must have significant violations
        const violationsCount = result.violations_summary?.violations_count || {};
        const totalViolations = Object.values(violationsCount).reduce((sum, count) => sum + count, 0);
        if (totalViolations === 0) {
            return false;
        }
        
        // All checks passed
        return true;
    }
    
    /**
     * Enhanced violation session saving with strict validation
     */
    async saveViolationSession(currentTime, result) {
        const sessionDuration = Math.round((currentTime - this.currentViolationSession.startTime) / 1000);
        
        // Create session fingerprint
        const violationsCount = result.violations_summary?.violations_count || {};
        const sessionFingerprint = JSON.stringify({
            types: Array.from(this.currentViolationSession.violationTypes).sort(),
            persons: result.violations_summary?.total_persons || 0,
            duration_range: Math.floor(sessionDuration / 10) * 10 // Group by 10-second ranges
        });
        
        // Check for duplicate sessions
        if (this.recentSessionFingerprints.has(sessionFingerprint)) {
            console.log('🚫 Duplicate session detected, skipping save');
            this.resetViolationSession();
            return;
        }
        
        const violationData = {
            timestamp: this.currentViolationSession.start,
            frame_count: this.frameCount,
            violation_session_start: this.currentViolationSession.start,
            violation_frames_count: this.violationFrameCount,
            violations_summary: result.violations_summary,
            persons: result.persons.filter(p => p.violations.length > 0),
            session_duration_seconds: sessionDuration,
            session_fingerprint: sessionFingerprint,
            violation_types_in_session: Array.from(this.currentViolationSession.violationTypes)
        };
        
        console.log('💾 Saving enhanced violation session:', {
            session_id: sessionFingerprint.substring(0, 8),
            duration: sessionDuration + 's',
            frames: this.violationFrameCount,
            types: Array.from(this.currentViolationSession.violationTypes)
        });
        
        // Save violation record
        await this.saveViolationRecord(violationData);
        
        // Track this session to prevent duplicates
        this.recentSessionFingerprints.add(sessionFingerprint);
        
        // Clean old fingerprints (keep only last 10)
        if (this.recentSessionFingerprints.size > 10) {
            const fingerprintArray = Array.from(this.recentSessionFingerprints);
            this.recentSessionFingerprints = new Set(fingerprintArray.slice(-10));
        }
        
        // Show enhanced notification
        const violationTypes = Array.from(this.currentViolationSession.violationTypes).join(', ');
        this.showNotification(
            `🚨 Violation session saved: ${sessionDuration}s, ${violationTypes}`,
            'warning'
        );
        
        this.resetViolationSession();
    }
    
    /**
     * Reset violation session tracking
     */
    resetViolationSession() {
        this.lastViolationSave = Date.now();
        this.violationFrameCount = 0;
        this.currentViolationSession = null;
    }
    
    /**
     * Handle when no violations are detected (session cleanup)
     */
    handleNoViolations() {
        // If we have an active session and it meets criteria, save it
        if (this.currentViolationSession) {
            const currentTime = Date.now();
            const sessionDuration = currentTime - this.currentViolationSession.startTime;
            
            // Only save if session was long enough and had enough violation frames
            if (sessionDuration >= this.minSessionDuration && 
                this.violationFrameCount >= this.minViolationFrames &&
                currentTime - this.lastViolationSave >= this.violationSaveInterval) {
                
                console.log('🔚 Ending violation session, will save on next detection cycle');
                // Don't save immediately here to avoid breaking detection flow
            } else {
                console.log('🗑️ Discarding short violation session:', {
                    duration: Math.round(sessionDuration / 1000) + 's',
                    frames: this.violationFrameCount,
                    required_duration: this.minSessionDuration / 1000 + 's',
                    required_frames: this.minViolationFrames
                });
                this.resetViolationSession();
            }
        }
    }
    
    /**
     * Save violation record to local storage and optionally to server
     */
    async saveViolationRecord(violationData) {
        try {
            // Save to local storage
            const existingRecords = JSON.parse(localStorage.getItem('ppe_violations') || '[]');
            existingRecords.push(violationData);
            
            // Keep only last 100 records in localStorage
            if (existingRecords.length > 100) {
                existingRecords.splice(0, existingRecords.length - 100);
            }
            
            localStorage.setItem('ppe_violations', JSON.stringify(existingRecords));
            
            // Also save to server if auto-save is enabled
            if (this.settings.autoSaveViolations) {
                try {
                    const response = await fetch('/violations/save', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(violationData)
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        console.log('✅ Violation saved to server:', result.message);
                        this.showNotification('Нарушение сохранено на сервер', 'success');
                    } else {
                        console.error('Failed to save violation to server:', response.statusText);
                    }
                } catch (serverError) {
                    console.warn('Server not available, violation saved locally only:', serverError);
                }
            }
        } catch (error) {
            console.error('Failed to save violation record:', error);
        }
    }
    
    /**
     * Update connection status indicator
     */
    updateConnectionStatus(status) {
        this.connectionStatus.className = `status ${status}`;
        this.connectionStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    /**
     * Toggle settings panel
     */
    toggleSettings() {
        this.settingsPanel.classList.toggle('open');
    }
    
    /**
     * Close settings panel
     */
    closeSettings() {
        this.settingsPanel.classList.remove('open');
    }
    
    /**
     * Handle keyboard shortcuts
     */
    handleKeyboard(event) {
        // Prevent shortcuts when typing in inputs
        if (event.target.tagName === 'INPUT') return;
        
        switch (event.code) {
            case 'Space':
                event.preventDefault();
                if (this.isDetectionActive) {
                    this.stopDetection();
                } else if (this.stream) {
                    this.startDetection();
                }
                break;
            case 'KeyC':
                if (!this.stream) {
                    this.startCamera();
                } else {
                    this.stopCamera();
                }
                break;
            case 'KeyS':
                this.toggleSettings();
                break;
            case 'Escape':
                this.closeSettings();
                break;
        }
    }
    
    /**
     * Handle window resize
     */
    handleResize() {
        // Adjust video container if needed
        if (window.innerWidth <= 768) {
            this.closeSettings();
        }
    }
    
    /**
     * Show notification to user
     */
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        this.notificationArea.appendChild(notification);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, duration);
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.isDetectionActive) {
            this.stopDetection();
        }
        if (this.stream) {
            this.stopCamera();
        }
    }
}

// Additional utility functions

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

/**
 * Download violation records as JSON
 */
function downloadViolationRecords() {
    try {
        const records = JSON.parse(localStorage.getItem('ppe_violations') || '[]');
        if (records.length === 0) {
            alert('No violation records to download');
            return;
        }
        
        const blob = new Blob([JSON.stringify(records, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `ppe_violations_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Failed to download violation records:', error);
        alert('Failed to download violation records');
    }
}

/**
 * Clear violation records
 */
function clearViolationRecords() {
    if (confirm('Are you sure you want to clear all violation records?')) {
        localStorage.removeItem('ppe_violations');
        alert('Violation records cleared');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🏗️ DOMContentLoaded event triggered');
    
    // Check for required browser features
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support camera access. Please use a modern browser.');
        return;
    }
    
    // Add additional debugging for button
    const btn = document.getElementById('startCameraBtn');
    console.log('🔍 Button check on DOMContentLoaded:', {
        found: !!btn,
        id: btn?.id,
        text: btn?.textContent?.trim(),
        disabled: btn?.disabled,
        display: btn ? window.getComputedStyle(btn).display : 'N/A',
        visibility: btn ? window.getComputedStyle(btn).visibility : 'N/A'
    });
    
    // Initialize the PPE detection app
    window.ppeApp = new PPEDetectionApp();
    
    // Initialize tab navigation
    initializeTabs();
    
    console.log('🚀 PPE Detection Web Interface loaded successfully');
    
    // Additional manual button test after a delay
    setTimeout(() => {
        console.log('🧪 Manual button test after 3 seconds...');
        const testBtn = document.getElementById('startCameraBtn');
        if (testBtn) {
            console.log('🔴 Manually clicking button for test...');
            testBtn.click();
        } else {
            console.error('❌ Button disappeared!');
        }
    }, 3000);
});

// Tab navigation functionality
function initializeTabs() {
    console.log('🗂️ Initializing tab navigation...');
    
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    console.log(`Found ${tabButtons.length} tab buttons and ${tabPanes.length} tab panes`);
    
    // Debug CSS issues
    tabButtons.forEach((button, index) => {
        const computedStyle = window.getComputedStyle(button);
        console.log(`Tab button ${index} CSS:`, {
            display: computedStyle.display,
            visibility: computedStyle.visibility,
            pointerEvents: computedStyle.pointerEvents,
            zIndex: computedStyle.zIndex,
            position: computedStyle.position
        });
    });
    
    // Add click event listeners to tab buttons
    tabButtons.forEach((button, index) => {
        console.log(`Tab button ${index}:`, {
            element: button,
            id: button.id,
            dataTab: button.dataset.tab,
            text: button.textContent.trim(),
            classList: button.classList.toString()
        });
        
        button.addEventListener('click', (event) => {
            console.log(`🗂️ Tab button clicked:`, {
                target: event.target,
                id: event.target.id,
                dataTab: event.target.dataset.tab
            });
            
            const targetTab = button.dataset.tab;
            console.log(`🗂️ Switching to tab: ${targetTab}`);
            switchTab(targetTab);
        });
        
        console.log(`✅ Event listener attached to tab: ${button.dataset.tab}`);
    });
    
    // Check system status for About tab
    checkSystemStatus();
}

function switchTab(targetTab) {
    console.log(`🔄 Switching to tab: ${targetTab}`);
    
    // Remove active class from all buttons and panes
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    
    // Add active class to target button and pane
    const targetButton = document.querySelector(`[data-tab="${targetTab}"]`);
    const targetPane = document.getElementById(`${targetTab}Content`);
    
    if (targetButton && targetPane) {
        targetButton.classList.add('active');
        targetPane.classList.add('active');
        console.log(`✅ Successfully switched to ${targetTab} tab`);
    } else {
        console.error(`❌ Failed to find tab elements for: ${targetTab}`);
    }
    
    // Trigger specific tab actions
    handleTabSwitch(targetTab);
}

function handleTabSwitch(tabName) {
    switch (tabName) {
        case 'camera':
            console.log('📹 Camera tab activated');
            // Camera tab specific actions if needed
            break;
            
        case 'docs':
            console.log('📖 Documentation tab activated');
            // Documentation tab specific actions if needed
            break;
            
        case 'about':
            console.log('ℹ️ About tab activated');
            // Update system status when About tab is viewed
            checkSystemStatus();
            break;
    }
}

function toggleIframe(iframeId) {
    const iframe = document.getElementById(iframeId);
    if (iframe) {
        if (iframe.style.display === 'none' || !iframe.style.display) {
            iframe.style.display = 'block';
            console.log(`📱 Showing inline iframe: ${iframeId}`);
        } else {
            iframe.style.display = 'none';
            console.log(`📱 Hiding inline iframe: ${iframeId}`);
        }
    }
}

async function checkSystemStatus() {
    console.log('🔍 Checking system status...');
    const statusElement = document.getElementById('systemStatus');
    
    if (!statusElement) return;
    
    try {
        statusElement.textContent = 'Checking...';
        
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusElement.innerHTML = '<span style="color: #2ecc71;">✅ Online & Healthy</span>';
        } else {
            statusElement.innerHTML = '<span style="color: #f39c12;">⚠️ Partial Service</span>';
        }
    } catch (error) {
        console.error('Failed to check system status:', error);
        statusElement.innerHTML = '<span style="color: #e74c3c;">❌ Offline</span>';
    }
}

// Manual test function for debugging tab issues
function testTabFunctionality() {
    console.log('🧪 Testing tab functionality manually...');
    
    const tabButtons = document.querySelectorAll('.tab-button');
    console.log('Found tab buttons:', tabButtons.length);
    
    tabButtons.forEach((button, index) => {
        console.log(`Tab ${index}:`, {
            id: button.id,
            dataTab: button.dataset.tab,
            text: button.textContent.trim(),
            disabled: button.disabled,
            clickable: window.getComputedStyle(button).pointerEvents !== 'none'
        });
    });
    
    // Test switching to docs tab
    console.log('Testing docs tab switch...');
    switchTab('docs');
    
    setTimeout(() => {
        console.log('Testing about tab switch...');
        switchTab('about');
        
        setTimeout(() => {
            console.log('Testing camera tab switch...');
            switchTab('camera');
        }, 1000);
    }, 1000);
}

// Violations Folder Management
class ViolationsFolderManager {
    constructor() {
        this.modal = document.getElementById('folderModal');
        this.folderStats = document.getElementById('folderStats');
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        console.log('🔧 Initializing ViolationsFolderManager event listeners...');
        
        // Open violations folder button
        const openViolationsFolderBtn = document.getElementById('openViolationsFolderBtn');
        console.log('📁 Violations folder button found:', !!openViolationsFolderBtn);
        
        if (openViolationsFolderBtn) {
            console.log('📁 Button element:', openViolationsFolderBtn);
            console.log('📁 Button computed style:', window.getComputedStyle(openViolationsFolderBtn));
            console.log('📁 Button disabled:', openViolationsFolderBtn.disabled);
            console.log('📁 Button pointer-events:', window.getComputedStyle(openViolationsFolderBtn).pointerEvents);
            console.log('📁 Button z-index:', window.getComputedStyle(openViolationsFolderBtn).zIndex);
            
            // Add multiple event listeners for debugging
            openViolationsFolderBtn.addEventListener('click', (e) => {
                console.log('📁 VIOLATIONS FOLDER BUTTON CLICKED!');
                console.log('📁 Event:', e);
                e.preventDefault();
                e.stopPropagation();
                this.showModal();
            });
            
            // Add mousedown for extra debugging
            openViolationsFolderBtn.addEventListener('mousedown', () => {
                console.log('📁 Violations folder button mousedown detected');
            });
            
            // Add mouseover for extra debugging
            openViolationsFolderBtn.addEventListener('mouseover', () => {
                console.log('📁 Violations folder button mouseover detected');
            });
            
            console.log('✅ Violations folder button event listeners attached');
        } else {
            console.error('❌ Violations folder button NOT found!');
        }

        // Modal action buttons
        const openFolderBtn = document.getElementById('openFolderBtn');
        const downloadArchiveBtn = document.getElementById('downloadArchiveBtn');
        const clearViolationsBtn = document.getElementById('clearViolationsBtn');
        const closeFolderModal = document.getElementById('closeFolderModal');

        if (openFolderBtn) {
            openFolderBtn.addEventListener('click', () => this.openViolationsFolder());
        }

        if (downloadArchiveBtn) {
            downloadArchiveBtn.addEventListener('click', () => this.downloadViolationsArchive());
        }

        if (clearViolationsBtn) {
            clearViolationsBtn.addEventListener('click', () => this.clearViolationsFolder());
        }

        if (closeFolderModal) {
            closeFolderModal.addEventListener('click', () => this.hideModal());
        }

        // Close modal when clicking outside
        if (this.modal) {
            this.modal.addEventListener('click', (e) => {
                if (e.target === this.modal) {
                    this.hideModal();
                }
            });
        }
    }

    async showModal() {
        console.log('📁 Opening violations folder modal...');
        if (this.modal) {
            this.modal.style.display = 'flex';
            await this.loadFolderStats();
        }
    }

    hideModal() {
        console.log('📁 Closing violations folder modal...');
        if (this.modal) {
            this.modal.style.display = 'none';
        }
    }

    async loadFolderStats() {
        if (!this.folderStats) return;

        this.folderStats.textContent = 'Загрузка статистики...';

        try {
            const response = await fetch('/violations/stats');
            if (response.ok) {
                const stats = await response.json();
                this.displayStats(stats);
            } else {
                // Fallback to local stats if API not available
                this.displayLocalStats();
            }
        } catch (error) {
            console.warn('API not available, using local stats:', error);
            this.displayLocalStats();
        }
    }

    displayStats(stats) {
        const {
            video_files = 0,
            json_files = 0,
            csv_files = 0,
            total_size = '0 MB',
            last_violation = 'Никогда'
        } = stats;

        this.folderStats.innerHTML = `
            📹 Видео: ${video_files} файлов<br>
            📄 JSON: ${json_files} отчетов<br>
            📊 CSV: ${csv_files} сводок<br>
            💾 Размер: ${total_size}<br>
            🕒 Последнее нарушение: ${last_violation}
        `;
    }

    displayLocalStats() {
        const violationRecords = JSON.parse(localStorage.getItem('ppe_violations') || '[]');
        const lastViolation = violationRecords.length > 0 
            ? new Date(violationRecords[violationRecords.length - 1].timestamp).toLocaleDateString('ru-RU')
            : 'Никогда';

        this.folderStats.innerHTML = `
            📹 Видео: Недоступно<br>
            📄 JSON: ${violationRecords.length} записей в браузере<br>
            📊 CSV: Недоступно<br>
            💾 Размер: Недоступно<br>
            🕒 Последнее нарушение: ${lastViolation}
        `;
    }

    async openViolationsFolder() {
        console.log('🗂️ Opening violations folder...');

        try {
            const response = await fetch('/violations/open-folder', {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message || 'Папка нарушений открыта', 'success');
            } else {
                throw new Error('Сервер не может открыть папку');
            }
        } catch (error) {
            console.error('Failed to open violations folder:', error);
            this.showNotification(
                'Не удалось открыть папку нарушений. Проверьте, что сервер запущен.',
                'error'
            );
        }
    }

    async downloadViolationsArchive() {
        console.log('📦 Downloading violations archive...');

        try {
            const response = await fetch('/violations/download-archive');

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `violations_archive_${new Date().toISOString().split('T')[0]}.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                this.showNotification('Архив нарушений скачан', 'success');
            } else {
                throw new Error('Сервер не может создать архив');
            }
        } catch (error) {
            console.error('Failed to download violations archive:', error);
            // Fallback to local data download
            this.downloadLocalViolations();
        }
    }

    downloadLocalViolations() {
        try {
            const records = JSON.parse(localStorage.getItem('ppe_violations') || '[]');
            if (records.length === 0) {
                this.showNotification('Нет записей нарушений для скачивания', 'warning');
                return;
            }

            const blob = new Blob([JSON.stringify(records, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `local_violations_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            URL.revokeObjectURL(url);
            this.showNotification('Локальные записи нарушений скачаны', 'success');
        } catch (error) {
            console.error('Failed to download local violations:', error);
            this.showNotification('Ошибка при скачивании записей', 'error');
        }
    }

    async clearViolationsFolder() {
        const confirmed = confirm(
            'Вы уверены, что хотите очистить папку нарушений?\n' +
            'Это действие нельзя отменить и удалит все записи видео и отчеты.'
        );

        if (!confirmed) return;

        console.log('🗑️ Clearing violations folder...');

        try {
            const response = await fetch('/violations/clear', {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message || 'Папка нарушений очищена', 'success');
                // Refresh stats
                await this.loadFolderStats();
            } else {
                throw new Error('Сервер не может очистить папку');
            }
        } catch (error) {
            console.error('Failed to clear violations folder:', error);
            // Fallback to clearing local data
            this.clearLocalViolations();
        }
    }

    clearLocalViolations() {
        try {
            localStorage.removeItem('ppe_violations');
            this.showNotification('Локальные записи нарушений очищены', 'success');
            this.loadFolderStats(); // Refresh stats
        } catch (error) {
            console.error('Failed to clear local violations:', error);
            this.showNotification('Ошибка при очистке записей', 'error');
        }
    }

    showNotification(message, type = 'info') {
        if (window.ppeApp && window.ppeApp.showNotification) {
            window.ppeApp.showNotification(message, type);
        } else {
            // Fallback notification
            alert(message);
        }
    }
}

// Initialize violations folder manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('📁 Initializing violations folder manager...');
    window.violationsFolderManager = new ViolationsFolderManager();
    console.log('📁 Violations folder manager initialized:', window.violationsFolderManager);
});

// Manual test function for violations folder button
function testViolationsFolderButton() {
    console.log('🧪 Testing violations folder button manually...');
    
    const button = document.getElementById('openViolationsFolderBtn');
    console.log('Button found:', !!button);
    
    if (button) {
        console.log('Button element:', button);
        console.log('Button disabled:', button.disabled);
        console.log('Button style display:', window.getComputedStyle(button).display);
        console.log('Button style visibility:', window.getComputedStyle(button).visibility);
        console.log('Button style pointer-events:', window.getComputedStyle(button).pointerEvents);
        console.log('Button style z-index:', window.getComputedStyle(button).zIndex);
        console.log('Button classList:', button.classList.toString());
        
        // Test manual click
        console.log('🔴 Manually clicking button for test...');
        button.click();
        
        // Test showing modal directly
        if (window.violationsFolderManager) {
            console.log('🔵 Testing direct modal show...');
            window.violationsFolderManager.showModal();
        }
    } else {
        console.error('❌ Button not found!');
    }
    
    // Check if modal exists
    const modal = document.getElementById('folderModal');
    console.log('Modal found:', !!modal);
    if (modal) {
        console.log('Modal style display:', window.getComputedStyle(modal).display);
    }
}

