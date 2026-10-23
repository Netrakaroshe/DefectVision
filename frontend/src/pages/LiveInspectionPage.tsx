import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Upload,
  CheckCircle2,
  AlertOctagon,
  AlertTriangle,
  Loader2,
  Image as ImageIcon,
  Sliders,
  Cpu,
  Layers,
  Clock,
  RotateCcw,
  Tag,
  Hash,
  Camera,
  CameraOff,
  Sparkles,
  Crosshair,
  ShieldCheck,
  Info,
} from 'lucide-react';
import { getProducts, inspectProduct, getInspectionHeatmapUrl, detectDents } from '../services/api';
import { Product, Inspection, DentDetectionResult } from '../types';

interface LiveInspectionPageProps {
  onNavigateToProducts?: () => void;
}

export const LiveInspectionPage: React.FC<LiveInspectionPageProps> = ({ onNavigateToProducts }) => {
  // Pipeline mode: Production Two-Stage vs Experimental Dent Detection
  const [pipelineMode, setPipelineMode] = useState<'production' | 'dent'>('production');

  // Production State
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Shared Acquisition mode: Live Camera or File Upload
  const [mode, setMode] = useState<'camera' | 'upload'>('camera');

  // Shared Camera state
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraLoading, setCameraLoading] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoReady, setVideoReady] = useState<boolean>(false);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [loadingProducts, setLoadingProducts] = useState<boolean>(true);
  const [inspecting, setInspecting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [inspectionResult, setInspectionResult] = useState<Inspection | null>(null);

  // Experimental Dent Detection state
  const [dentConfidenceThreshold, setDentConfidenceThreshold] = useState<number>(0.30);
  const [dentResult, setDentResult] = useState<DentDetectionResult | null>(null);
  const [dentResultImageUrl, setDentResultImageUrl] = useState<string | null>(null);

  // Generate preview URL for the currently selected file (upload mode)
  const filePreviewUrl = useMemo(() => {
    if (!selectedFile) return null;
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  // Clean up object URL when file changes or component unmounts
  useEffect(() => {
    return () => {
      if (filePreviewUrl) {
        URL.revokeObjectURL(filePreviewUrl);
      }
    };
  }, [filePreviewUrl]);

  // Stop camera media tracks helper
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore error on stop
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setVideoReady(false);
    setVideoDimensions(null);
  };

  // Clean up camera on component unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Load products on mount
  useEffect(() => {
    async function loadProducts() {
      setLoadingProducts(true);
      setError(null);
      try {
        const allProducts = await getProducts();
        setProducts(allProducts);
        // Default to first READY product if available
        const firstReady = allProducts.find((p) => p.status === 'READY');
        if (firstReady) {
          setSelectedProductId(firstReady.product_id);
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to load products list.');
      } finally {
        setLoadingProducts(false);
      }
    }
    loadProducts();
  }, []);

  // Filter only READY products with an active calibration
  const readyProducts = useMemo(() => {
    return products.filter((p) => p.status === 'READY' && p.active_calibration != null);
  }, [products]);

  const selectedProduct = useMemo(() => {
    return readyProducts.find((p) => p.product_id === selectedProductId) || null;
  }, [readyProducts, selectedProductId]);

  // Start camera stream
  const startCamera = async () => {
    setCameraError(null);
    setError(null);
    setCameraLoading(true);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          'Webcam access is not supported by your browser or environment. Please ensure you are accessing via localhost or HTTPS.'
        );
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment',
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (playErr) {
          console.warn('Video play interrupted or delayed:', playErr);
        }
      }
      setIsCameraActive(true);
    } catch (err: unknown) {
      let msg = 'Failed to access camera.';
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          msg = 'Camera permission denied. Please grant camera access in your browser settings to inspect live frames.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          msg = 'No video capture device found. Please connect a webcam or switch to File Upload mode.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          msg = 'Camera is currently in use by another application or the operating system locked the device.';
        } else if (err.name === 'OverconstrainedError') {
          msg = 'Camera resolution constraints could not be satisfied by the available video hardware.';
        } else {
          msg = `Camera error (${err.name}): ${err.message}`;
        }
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setCameraError(msg);
      stopCamera();
    } finally {
      setCameraLoading(false);
    }
  };

  const handleVideoLoadedMetadata = () => {
    if (videoRef.current && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
      setVideoDimensions({
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight,
      });
      setVideoReady(true);
    }
  };

  const handlePipelineModeChange = (newPipeline: 'production' | 'dent') => {
    if (inspecting) return;
    setPipelineMode(newPipeline);
    setError(null);
  };

  const handleModeChange = (newMode: 'camera' | 'upload') => {
    if (inspecting) return;
    if (newMode === 'upload' && isCameraActive) {
      stopCamera();
    }
    setMode(newMode);
    setError(null);
    setCameraError(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    if (file.type.startsWith('image/')) {
      setSelectedFile(file);
      setInspectionResult(null);
      setDentResult(null);
      setDentResultImageUrl(null);
      setError(null);
    } else {
      setError('Please select a valid image file (PNG, JPG, WEBP, BMP).');
    }
  };

  const handleClearImage = () => {
    if (inspecting) return;
    setSelectedFile(null);
    setInspectionResult(null);
    setDentResult(null);
    setDentResultImageUrl(null);
    setError(null);
  };

  const handleResetInspection = () => {
    if (inspecting) return;
    setInspectionResult(null);
    setError(null);
    if (mode === 'upload') {
      setSelectedFile(null);
    }
  };

  const handleResetDent = () => {
    if (inspecting) return;
    setDentResult(null);
    setDentResultImageUrl(null);
    setError(null);
    if (mode === 'upload') {
      setSelectedFile(null);
    }
  };

  // Inspect uploaded file (Production pipeline)
  const handleRunUploadInspection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedProduct) {
      setError('Please select a READY product to inspect.');
      return;
    }
    if (!selectedFile) {
      setError('Please select an image file to inspect.');
      return;
    }

    setInspecting(true);
    setError(null);

    try {
      const result = await inspectProduct(selectedProduct.product_id, selectedFile, true);
      setInspectionResult(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Inspection execution failed on the backend.');
    } finally {
      setInspecting(false);
    }
  };

  // Inspect live camera frame (Production pipeline)
  const handleInspectCameraFrame = async () => {
    if (!selectedProduct) {
      setError('Please select a READY product before inspecting.');
      return;
    }
    if (!isCameraActive || !videoRef.current || !videoReady) {
      setError('Camera feed is not ready. Start the camera and wait for video transmission before capturing.');
      return;
    }
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight || video.videoWidth === 0 || video.videoHeight === 0) {
      setError('Camera video feed has invalid dimensions. Please ensure camera is active and transmitting.');
      return;
    }

    setInspecting(true);
    setError(null);

    try {
      const canvas = canvasRef.current || document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to acquire 2D canvas context for frame capture.');
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.95);
      });

      if (!blob) {
        throw new Error('Failed to encode captured video frame to JPEG format.');
      }

      const result = await inspectProduct(selectedProduct.product_id, blob, true);
      setInspectionResult(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Camera frame inspection failed on backend.');
    } finally {
      setInspecting(false);
    }
  };

  // Run Dent Detection on uploaded file
  const handleRunDentUploadInspection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select an image file to inspect for surface dents.');
      return;
    }

    setInspecting(true);
    setError(null);

    try {
      const result = await detectDents(selectedFile, dentConfidenceThreshold);
      setDentResult(result);
      if (filePreviewUrl) {
        setDentResultImageUrl(filePreviewUrl);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Dent detection inference failed on backend.');
    } finally {
      setInspecting(false);
    }
  };

  // Run Dent Detection on live camera frame
  const handleInspectDentCameraFrame = async () => {
    if (!isCameraActive || !videoRef.current || !videoReady) {
      setError('Camera feed is not ready. Start the camera and wait for video stream before capturing.');
      return;
    }
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight || video.videoWidth === 0 || video.videoHeight === 0) {
      setError('Camera video feed has invalid dimensions. Please ensure camera is active.');
      return;
    }

    setInspecting(true);
    setError(null);

    try {
      const canvas = canvasRef.current || document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to acquire 2D canvas context for frame capture.');
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const capturedDataUrl = canvas.toDataURL('image/jpeg', 0.95);

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.95);
      });

      if (!blob) {
        throw new Error('Failed to encode captured video frame to JPEG format.');
      }

      const result = await detectDents(blob, dentConfidenceThreshold);
      setDentResult(result);
      setDentResultImageUrl(capturedDataUrl);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Dent detection inference failed on backend.');
    } finally {
      setInspecting(false);
    }
  };

  const isNormal = inspectionResult?.verdict === 'NORMAL';

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-industrial-900/80 border border-industrial-800 rounded-xl p-5">
        <h1 className="text-base font-bold text-white tracking-wide">Live Visual Inspection Station</h1>
        <p className="text-xs text-industrial-400 mt-1 max-w-2xl">
          Real-time industrial defect inspection station with two-stage PatchCore + ResNet-18 pipeline and dedicated experimental Dent detection.
        </p>
      </div>

      {/* Pipeline Task Selector Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-industrial-900 border border-industrial-800 rounded-xl p-3">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono text-industrial-400 font-semibold uppercase tracking-wider pl-1">
            Inspection Pipeline:
          </span>
          <div className="flex items-center space-x-1 bg-industrial-950 p-1 rounded-lg border border-industrial-800">
            <button
              type="button"
              onClick={() => handlePipelineModeChange('production')}
              disabled={inspecting}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-mono transition-colors ${
                pipelineMode === 'production'
                  ? 'bg-sky-600 text-white font-semibold shadow'
                  : 'text-industrial-400 hover:text-white'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Production Two-Stage</span>
            </button>

            <button
              type="button"
              onClick={() => handlePipelineModeChange('dent')}
              disabled={inspecting}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-mono transition-colors ${
                pipelineMode === 'dent'
                  ? 'bg-amber-600 text-white font-semibold shadow'
                  : 'text-industrial-400 hover:text-white'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-300" />
              <span>Experimental Dent Detection</span>
              <span className="px-1.5 py-0.2 text-[9px] rounded bg-amber-400/20 text-amber-300 border border-amber-400/40 uppercase font-mono font-bold tracking-tight">
                CR7-DET
              </span>
            </button>
          </div>
        </div>

        {pipelineMode === 'dent' && (
          <div className="flex items-center space-x-1.5 text-[11px] font-mono text-amber-300/90 bg-amber-950/40 border border-amber-800/60 px-3 py-1 rounded-lg">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span>Faster R-CNN MobileNetV3 • Isolated from Production Verdicts</span>
          </div>
        )}
      </div>

      {/* Error alert banner */}
      {error && (
        <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-xs text-red-300 flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Inspection Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Product Selection / Model Config & Image Input */}
        <div className="lg:col-span-5 space-y-5">
          {/* Card 1: PRODUCTION MODE - Product Selector */}
          {pipelineMode === 'production' && (
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center space-x-2 border-b border-industrial-800 pb-3">
                <Sliders className="w-4 h-4 text-sky-400" />
                <h2 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                  1. Select READY Product
                </h2>
              </div>

              {loadingProducts ? (
                <div className="py-4 text-center text-xs text-industrial-400 font-mono space-y-2">
                  <Loader2 className="w-4 h-4 animate-spin mx-auto text-sky-400" />
                  <div>Loading products...</div>
                </div>
              ) : readyProducts.length === 0 ? (
                <div className="bg-amber-950/40 border border-amber-800/80 rounded-lg p-4 space-y-2 text-xs text-amber-300">
                  <div className="flex items-center space-x-2 font-semibold text-amber-400">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>No Calibrated Products Available</span>
                  </div>
                  <p className="text-[11px] text-industrial-300">
                    Inspection requires a product with an active anomaly threshold calibration. Please calibrate a product in the{' '}
                    <span className="text-sky-300 font-semibold">Products</span> tab first.
                  </p>
                  {onNavigateToProducts && (
                    <button
                      type="button"
                      onClick={onNavigateToProducts}
                      className="mt-2 inline-flex items-center px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-mono font-medium transition-colors"
                    >
                      Go to Products
                    </button>
                  )}
                </div>
              ) : (
                <div className="space-y-3">
                  <div>
                    <label htmlFor="product-select" className="block text-xs font-mono text-industrial-300 mb-1">
                      Calibrated Product
                    </label>
                    <select
                      id="product-select"
                      value={selectedProductId}
                      onChange={(e) => {
                        setSelectedProductId(e.target.value);
                        setInspectionResult(null);
                        setError(null);
                      }}
                      disabled={inspecting}
                      className="w-full bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-sky-500"
                    >
                      {readyProducts.map((p) => (
                        <option key={p.product_id} value={p.product_id}>
                          {p.name} ({p.product_id})
                        </option>
                      ))}
                    </select>
                  </div>

                  {selectedProduct && selectedProduct.active_calibration && (
                    <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 space-y-1.5 font-mono text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-industrial-400">Active Threshold:</span>
                        <span className="text-emerald-400 font-bold">
                          {selectedProduct.active_calibration.threshold.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-industrial-400">Calibration Method:</span>
                        <span className="text-white capitalize">
                          {selectedProduct.active_calibration.calibration_method}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-industrial-400">Image Resolution:</span>
                        <span className="text-white">
                          {selectedProduct.active_calibration.image_width} &times;{' '}
                          {selectedProduct.active_calibration.image_height}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Card 1: DENT MODE - Model & Threshold Configuration */}
          {pipelineMode === 'dent' && (
            <div className="bg-industrial-900 border border-amber-900/60 rounded-xl p-5 space-y-4">
              <div className="flex items-center space-x-2 border-b border-industrial-800 pb-3">
                <Crosshair className="w-4 h-4 text-amber-400" />
                <h2 className="text-xs font-semibold text-amber-300 uppercase tracking-wider font-mono">
                  1. Dent Detector Configuration
                </h2>
              </div>

              {/* Model Specifications */}
              <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 space-y-1.5 font-mono text-[11px]">
                <div className="flex justify-between">
                  <span className="text-industrial-400">Architecture:</span>
                  <span className="text-white font-medium">Faster R-CNN MobileNetV3</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-industrial-400">Target Defect Class:</span>
                  <span className="text-amber-400 font-bold">Dents (Class ID 2)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-industrial-400">Validation Dataset:</span>
                  <span className="text-sky-300">CR7-DET (Steel Surface)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-industrial-400">Validation Dent AP@50:</span>
                  <span className="text-emerald-400 font-bold">89.3%</span>
                </div>
              </div>

              {/* Configurable Confidence Threshold Slider */}
              <div className="space-y-2 pt-1">
                <div className="flex items-center justify-between font-mono text-xs">
                  <label htmlFor="dent-thresh-slider" className="text-industrial-300 font-medium">
                    Confidence Threshold:
                  </label>
                  <span className="text-amber-400 font-bold px-2 py-0.5 rounded bg-amber-950/60 border border-amber-800">
                    {(dentConfidenceThreshold * 100).toFixed(0)}% ({dentConfidenceThreshold.toFixed(2)})
                  </span>
                </div>

                <input
                  id="dent-thresh-slider"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={dentConfidenceThreshold}
                  onChange={(e) => {
                    setDentConfidenceThreshold(parseFloat(e.target.value));
                    setDentResult(null);
                  }}
                  disabled={inspecting}
                  className="w-full accent-amber-500 cursor-pointer"
                />

                <div className="flex justify-between text-[10px] font-mono text-industrial-500">
                  <span>0.05 (High Recall)</span>
                  <span className="text-amber-400/80">0.30 (Pilot Default)</span>
                  <span>0.95 (High Precision)</span>
                </div>
              </div>

              {/* Scope & Limitation Disclaimer */}
              <div className="bg-amber-950/30 border border-amber-800/50 rounded-lg p-2.5 text-[11px] text-amber-200/90 flex items-start space-x-2">
                <Info className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                <span>
                  <strong>Experimental Limitation:</strong> Validated on CR7-DET steel surfaces only. Operates independently and does not assign a production verdict or modify PatchCore calibration.
                </span>
              </div>
            </div>
          )}

          {/* Card 2: Acquisition Mode & Source (Shared) */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
              <div className="flex items-center space-x-2">
                <ImageIcon className="w-4 h-4 text-sky-400" />
                <h2 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                  2. Image Acquisition
                </h2>
              </div>

              {/* Mode Selector Tabs */}
              <div className="flex items-center space-x-1 bg-industrial-950 p-1 rounded-lg border border-industrial-800">
                <button
                  type="button"
                  onClick={() => handleModeChange('camera')}
                  disabled={inspecting}
                  className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                    mode === 'camera'
                      ? pipelineMode === 'dent'
                        ? 'bg-amber-600 text-white font-semibold shadow'
                        : 'bg-sky-600 text-white font-semibold shadow'
                      : 'text-industrial-400 hover:text-white'
                  }`}
                >
                  <Camera className="w-3.5 h-3.5" />
                  <span>Live Camera</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleModeChange('upload')}
                  disabled={inspecting}
                  className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                    mode === 'upload'
                      ? pipelineMode === 'dent'
                        ? 'bg-amber-600 text-white font-semibold shadow'
                        : 'bg-sky-600 text-white font-semibold shadow'
                      : 'text-industrial-400 hover:text-white'
                  }`}
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>File Upload</span>
                </button>
              </div>
            </div>

            {/* Hidden canvas for video frame extraction */}
            <canvas ref={canvasRef} className="hidden" />

            {/* LIVE CAMERA MODE */}
            {mode === 'camera' && (
              <div className="space-y-4">
                {/* Video Viewport */}
                <div className="relative rounded-lg overflow-hidden border border-industrial-800 bg-black aspect-video flex items-center justify-center">
                  {/* Real video feed */}
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    onLoadedMetadata={handleVideoLoadedMetadata}
                    className={`w-full h-full object-contain ${isCameraActive ? 'block' : 'hidden'}`}
                  />

                  {/* Feed inactive placeholder */}
                  {!isCameraActive && (
                    <div className="text-center p-6 space-y-2">
                      <Camera className="w-8 h-8 text-industrial-600 mx-auto" />
                      <div className="text-xs font-mono text-industrial-400">Webcam Feed Inactive</div>
                      <p className="text-[11px] text-industrial-500 max-w-xs mx-auto">
                        Click <span className="text-sky-400 font-semibold">Start Camera</span> below to initialize the video feed for real-time inspection.
                      </p>
                    </div>
                  )}

                  {/* Overlays when camera is active */}
                  {isCameraActive && (
                    <>
                      {/* Top Badges */}
                      <div className="absolute top-2.5 left-2.5 flex items-center space-x-1.5 px-2 py-0.5 rounded bg-black/75 border border-industrial-700 text-[10px] font-mono text-emerald-400">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                        <span>LIVE</span>
                      </div>

                      {videoDimensions && (
                        <div className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded bg-black/75 border border-industrial-700 text-[10px] font-mono text-industrial-300">
                          {videoDimensions.width} &times; {videoDimensions.height}
                        </div>
                      )}

                      {/* Viewfinder Target Reticle */}
                      <div className="absolute inset-8 border border-white/20 pointer-events-none rounded">
                        <div className={`absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 ${pipelineMode === 'dent' ? 'border-amber-400' : 'border-sky-400'}`} />
                        <div className={`absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 ${pipelineMode === 'dent' ? 'border-amber-400' : 'border-sky-400'}`} />
                        <div className={`absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 ${pipelineMode === 'dent' ? 'border-amber-400' : 'border-sky-400'}`} />
                        <div className={`absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 ${pipelineMode === 'dent' ? 'border-amber-400' : 'border-sky-400'}`} />
                      </div>
                    </>
                  )}
                </div>

                {/* Camera Error Message */}
                {cameraError && (
                  <div className="bg-red-950/50 border border-red-800 rounded-lg p-3 text-xs text-red-300 flex items-start space-x-2">
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                    <div>
                      <div className="font-semibold text-red-200">Camera Startup Error</div>
                      <div className="text-[11px] text-red-300 mt-0.5">{cameraError}</div>
                    </div>
                  </div>
                )}

                {/* Camera Controls Bar */}
                <div className="flex items-center space-x-2">
                  {!isCameraActive ? (
                    <button
                      type="button"
                      onClick={startCamera}
                      disabled={cameraLoading || inspecting}
                      className="flex-1 flex items-center justify-center space-x-2 py-2.5 px-4 bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white rounded-lg text-xs font-semibold tracking-wide transition-colors shadow-md font-mono"
                    >
                      {cameraLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Initializing Camera...</span>
                        </>
                      ) : (
                        <>
                          <Camera className="w-4 h-4" />
                          <span>Start Camera</span>
                        </>
                      )}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={stopCamera}
                      disabled={inspecting}
                      className="flex items-center justify-center space-x-2 py-2 px-3 bg-industrial-950 hover:bg-industrial-800 text-industrial-300 hover:text-red-300 rounded-lg text-xs font-mono border border-industrial-700 transition-colors"
                    >
                      <CameraOff className="w-3.5 h-3.5" />
                      <span>Stop Camera</span>
                    </button>
                  )}

                  {/* ACTION BUTTON: Production vs Dent */}
                  {pipelineMode === 'production' ? (
                    <button
                      type="button"
                      onClick={handleInspectCameraFrame}
                      disabled={!selectedProduct || !isCameraActive || !videoReady || inspecting}
                      className="flex-1 flex items-center justify-center space-x-2 py-2.5 px-4 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded-lg text-xs font-semibold tracking-wide transition-colors shadow-md font-mono"
                    >
                      {inspecting ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Inspecting Frame...</span>
                        </>
                      ) : (
                        <>
                          <Cpu className="w-4 h-4" />
                          <span>Inspect Frame</span>
                        </>
                      )}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleInspectDentCameraFrame}
                      disabled={!isCameraActive || !videoReady || inspecting}
                      className="flex-1 flex items-center justify-center space-x-2 py-2.5 px-4 bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-white rounded-lg text-xs font-semibold tracking-wide transition-colors shadow-md font-mono"
                    >
                      {inspecting ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Detecting Dents...</span>
                        </>
                      ) : (
                        <>
                          <Crosshair className="w-4 h-4" />
                          <span>Detect Dents</span>
                        </>
                      )}
                    </button>
                  )}
                </div>

                {/* Camera Helper Note */}
                <p className="text-[11px] text-industrial-500 font-mono text-center">
                  {pipelineMode === 'production'
                    ? !selectedProduct
                      ? 'Select a READY product above to enable frame inspection.'
                      : !isCameraActive
                      ? 'Start camera stream to begin real-time frame inspection.'
                      : 'Aim sample within viewfinder and click Inspect Frame.'
                    : !isCameraActive
                    ? 'Start camera stream to begin Dent localization on steel surface.'
                    : 'Aim steel sample within viewfinder and click Detect Dents.'}
                </p>
              </div>
            )}

            {/* FILE UPLOAD MODE */}
            {mode === 'upload' && (
              <div className="space-y-4">
                {selectedFile && (
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={handleClearImage}
                      disabled={inspecting}
                      className="text-[11px] font-mono text-industrial-400 hover:text-red-400 transition-colors disabled:opacity-40"
                    >
                      Clear image
                    </button>
                  </div>
                )}

                {/* Dropzone */}
                {!selectedFile ? (
                  <div className="border-2 border-dashed border-industrial-700 hover:border-sky-500/60 rounded-xl p-6 text-center cursor-pointer transition-colors bg-industrial-950/50">
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
                      onChange={handleFileChange}
                      disabled={inspecting || (pipelineMode === 'production' && !selectedProduct)}
                      className="hidden"
                      id="inspection-image-input"
                    />
                    <label
                      htmlFor="inspection-image-input"
                      className={`block space-y-2 ${
                        (pipelineMode === 'production' && !selectedProduct) || inspecting
                          ? 'cursor-not-allowed opacity-50'
                          : 'cursor-pointer'
                      }`}
                    >
                      <Upload className="w-6 h-6 text-industrial-400 mx-auto" />
                      <div className="text-xs text-industrial-300">
                        {pipelineMode === 'production'
                          ? 'Click to select an inspection image from disk'
                          : 'Click to select a steel surface image to test for Dents'}
                      </div>
                      <div className="text-[11px] text-industrial-500 font-mono">
                        PNG, JPG, WEBP, BMP — One test sample
                      </div>
                    </label>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="relative rounded-lg overflow-hidden border border-industrial-800 bg-black aspect-video max-h-56 flex items-center justify-center">
                      {filePreviewUrl && (
                        <img
                          src={filePreviewUrl}
                          alt="Query sample"
                          className="max-h-full max-w-full object-contain mx-auto"
                        />
                      )}
                    </div>
                    <div className="flex items-center justify-between text-xs font-mono text-industrial-400 px-1">
                      <span className="truncate max-w-[240px] text-white font-medium">{selectedFile.name}</span>
                      <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                    </div>
                  </div>
                )}

                {/* Upload Inspect Button: Production vs Dent */}
                {pipelineMode === 'production' ? (
                  <form onSubmit={handleRunUploadInspection}>
                    <button
                      type="submit"
                      disabled={!selectedProduct || !selectedFile || inspecting}
                      className="w-full flex items-center justify-center space-x-2 py-2.5 px-4 bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white rounded-lg text-xs font-semibold tracking-wide transition-colors shadow-md font-mono"
                    >
                      {inspecting ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Executing Two-Stage Inspection...</span>
                        </>
                      ) : (
                        <>
                          <Cpu className="w-4 h-4" />
                          <span>Run Inspection</span>
                        </>
                      )}
                    </button>
                  </form>
                ) : (
                  <form onSubmit={handleRunDentUploadInspection}>
                    <button
                      type="submit"
                      disabled={!selectedFile || inspecting}
                      className="w-full flex items-center justify-center space-x-2 py-2.5 px-4 bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-white rounded-lg text-xs font-semibold tracking-wide transition-colors shadow-md font-mono"
                    >
                      {inspecting ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Detecting Dents...</span>
                        </>
                      ) : (
                        <>
                          <Crosshair className="w-4 h-4" />
                          <span>Detect Dents</span>
                        </>
                      )}
                    </button>
                  </form>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Inspection / Dent Results Display */}
        <div className="lg:col-span-7">
          {/* ============================================================== */}
          {/* DENT DETECTION RESULTS                                          */}
          {/* ============================================================== */}
          {pipelineMode === 'dent' ? (
            dentResult ? (
              <div className="bg-industrial-900 border border-amber-900/50 rounded-xl p-5 space-y-5">
                {/* Dent Detection Verdict Header */}
                <div
                  className={`p-4 rounded-xl border flex items-center justify-between ${
                    dentResult.dents_detected > 0
                      ? 'bg-amber-950/60 border-amber-800 text-amber-300'
                      : 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    {dentResult.dents_detected > 0 ? (
                      <AlertTriangle className="w-8 h-8 text-amber-400 shrink-0" />
                    ) : (
                      <ShieldCheck className="w-8 h-8 text-emerald-400 shrink-0" />
                    )}
                    <div>
                      <div className="text-xl font-bold tracking-wider font-mono">
                        {dentResult.dents_detected > 0
                          ? `${dentResult.dents_detected} DENT${dentResult.dents_detected > 1 ? 'S' : ''} DETECTED`
                          : 'NO DENTS DETECTED'}
                      </div>
                      <div className="text-xs text-industrial-300">
                        {dentResult.message}
                      </div>
                    </div>
                  </div>

                  <div className="text-right font-mono text-xs">
                    <div className="text-industrial-400 text-[10px]">Filter Threshold</div>
                    <div className="text-amber-400 font-bold">
                      &ge; {(dentResult.confidence_threshold * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                {/* Bounding Box Visualizer */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono text-industrial-300">
                    <span className="flex items-center space-x-2">
                      <Crosshair className="w-3.5 h-3.5 text-amber-400" />
                      <span>Dent Bounding Box Overlay</span>
                    </span>
                    <span className="text-[11px] text-industrial-400">
                      {dentResult.image_width} &times; {dentResult.image_height} px
                    </span>
                  </div>

                  <div className="relative rounded-lg overflow-hidden border border-industrial-800 bg-black aspect-square max-h-80 flex items-center justify-center">
                    {dentResultImageUrl && (
                      <img
                        src={dentResultImageUrl}
                        alt="Inspected steel surface"
                        className="w-full h-full object-contain"
                      />
                    )}

                    {/* SVG overlay rendering exact bounding boxes */}
                    <svg
                      viewBox={`0 0 ${dentResult.image_width} ${dentResult.image_height}`}
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{ objectFit: 'contain' }}
                    >
                      {dentResult.boxes.map((box, idx) => {
                        const boxW = Math.max(1, box.x2 - box.x1);
                        const boxH = Math.max(1, box.y2 - box.y1);
                        const labelY = Math.max(18, box.y1 - 6);
                        return (
                          <g key={idx}>
                            {/* Bounding rectangle */}
                            <rect
                              x={box.x1}
                              y={box.y1}
                              width={boxW}
                              height={boxH}
                              fill="rgba(245, 158, 11, 0.22)"
                              stroke="#f59e0b"
                              strokeWidth="3"
                              rx="2"
                            />
                            {/* Label backdrop badge */}
                            <rect
                              x={box.x1}
                              y={Math.max(2, box.y1 - 20)}
                              width={110}
                              height={18}
                              fill="#f59e0b"
                              rx="3"
                            />
                            {/* Text label */}
                            <text
                              x={box.x1 + 4}
                              y={labelY}
                              fill="#000"
                              fontSize="11"
                              fontWeight="bold"
                              fontFamily="monospace"
                            >
                              {`Dent ${(box.confidence * 100).toFixed(1)}%`}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                </div>

                {/* Detected Dents Breakdown List */}
                <div className="space-y-2">
                  <div className="text-xs font-mono font-semibold text-industrial-300">
                    Detection Coordinates & Confidence:
                  </div>

                  {dentResult.dents_detected === 0 ? (
                    <div className="p-3 bg-industrial-950 rounded-lg border border-industrial-800 text-xs text-industrial-400 font-mono text-center">
                      No candidate boxes met or exceeded the {(dentResult.confidence_threshold * 100).toFixed(0)}% threshold. Try lowering the threshold if inspecting faint indentations.
                    </div>
                  ) : (
                    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                      {dentResult.boxes.map((box, idx) => (
                        <div
                          key={idx}
                          className="bg-industrial-950 p-2.5 rounded-lg border border-industrial-800 flex items-center justify-between font-mono text-xs"
                        >
                          <div className="flex items-center space-x-2">
                            <span className="w-5 h-5 rounded bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-[11px]">
                              {idx + 1}
                            </span>
                            <span className="text-white font-medium">Dent</span>
                            <span className="text-industrial-500 text-[11px]">
                              [{box.x1}, {box.y1}] &rarr; [{box.x2}, {box.y2}]
                            </span>
                          </div>

                          <div className="flex items-center space-x-3">
                            <span className="text-[11px] text-industrial-400">
                              {Math.round(box.x2 - box.x1)} &times; {Math.round(box.y2 - box.y1)} px
                            </span>
                            <span className="text-amber-400 font-bold px-2 py-0.5 rounded bg-amber-950 border border-amber-800 text-xs">
                              {(box.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Traceability & Limitation Metadata */}
                <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3 text-[11px] font-mono text-industrial-400 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-industrial-400">Model Architecture:</span>
                    <span className="text-industrial-300">{dentResult.model_name}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-industrial-400">Validation Dataset:</span>
                    <span className="text-industrial-300">{dentResult.dataset}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-industrial-400">Inference Latency:</span>
                    <span className="text-amber-400 font-semibold">{dentResult.execution_time_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="pt-1 border-t border-industrial-900 text-[10px] text-industrial-500">
                    * Results are strictly isolated from production records and do not affect product calibration or historical metrics.
                  </div>
                </div>

                {/* Reset action */}
                <div className="flex justify-end pt-2">
                  <button
                    type="button"
                    onClick={handleResetDent}
                    className="flex items-center space-x-2 px-4 py-2 bg-industrial-800 hover:bg-industrial-700 text-white rounded-lg text-xs font-mono transition-colors border border-industrial-700"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Inspect Next Sample</span>
                  </button>
                </div>
              </div>
            ) : (
              /* Dent Mode Standby */
              <div className="bg-industrial-900 border border-amber-900/30 rounded-xl p-12 text-center space-y-3">
                <Sparkles className="w-10 h-10 text-amber-500/70 mx-auto" />
                <div className="text-sm font-semibold text-industrial-200">Experimental Dent Detection Station Standby</div>
                <p className="text-xs text-industrial-400 max-w-sm mx-auto">
                  Adjust the confidence threshold on the left, capture a webcam frame or upload a steel surface sample, and click{' '}
                  <span className="text-amber-400 font-semibold font-mono">Detect Dents</span> to localize surface indentations.
                </p>
                <div className="pt-2 text-[11px] font-mono text-amber-400/80">
                  Targeted Model: Faster R-CNN MobileNetV3-Large (CR7-DET Validated)
                </div>
              </div>
            )
          ) : (
            /* ============================================================== */
            /* PRODUCTION TWO-STAGE RESULTS                                   */
            /* ============================================================== */
            inspectionResult ? (
              <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-5">
                {/* Verdict Header */}
                <div
                  className={`p-4 rounded-xl border flex items-center justify-between ${
                    isNormal
                      ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                      : 'bg-rose-950/60 border-rose-800 text-rose-300'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    {isNormal ? (
                      <CheckCircle2 className="w-8 h-8 text-emerald-400 shrink-0" />
                    ) : (
                      <AlertOctagon className="w-8 h-8 text-rose-400 shrink-0" />
                    )}
                    <div>
                      <div className="text-xl font-bold tracking-wider font-mono">
                        {inspectionResult.verdict}
                      </div>
                      <div className="text-xs text-industrial-300">
                        {isNormal
                          ? 'Anomaly score is below the calibrated threshold. Sample passed inspection.'
                          : 'Anomaly score exceeds calibrated threshold. Defect detected.'}
                      </div>
                    </div>
                  </div>

                  <div className="text-right font-mono text-xs">
                    <div className="text-industrial-400 text-[10px]">Product</div>
                    <div className="text-white font-bold">{inspectionResult.product_id}</div>
                  </div>
                </div>

                {/* Two-Stage Architecture Breakdown */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Stage 1: PatchCore Anomaly Scoring */}
                  <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-4 space-y-3">
                    <div className="flex items-center space-x-2 text-xs font-mono font-semibold text-sky-400 border-b border-industrial-850 pb-2">
                      <Sliders className="w-3.5 h-3.5" />
                      <span>Stage 1: PatchCore Anomaly</span>
                    </div>

                    <div className="space-y-2 text-xs font-mono">
                      <div className="flex justify-between">
                        <span className="text-industrial-400">Anomaly Score:</span>
                        <span className="text-white font-bold">{inspectionResult.anomaly_score.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-industrial-400">Product Threshold:</span>
                        <span className="text-sky-300">{inspectionResult.anomaly_threshold.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-industrial-400">Margin:</span>
                        <span
                          className={
                            inspectionResult.margin > 0 ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'
                          }
                        >
                          {inspectionResult.margin > 0 ? `+${inspectionResult.margin.toFixed(4)}` : inspectionResult.margin.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between pt-1 border-t border-industrial-900">
                        <span className="text-industrial-400">Decision:</span>
                        <span className={isNormal ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                          {inspectionResult.verdict}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Stage 2: Supervised Defect Classification */}
                  <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-4 space-y-3">
                    <div className="flex items-center space-x-2 text-xs font-mono font-semibold text-sky-400 border-b border-industrial-850 pb-2">
                      <Tag className="w-3.5 h-3.5" />
                      <span>Stage 2: Defect Classifier</span>
                    </div>

                    {!inspectionResult.stage2_executed ? (
                      <div className="py-3 text-center space-y-1">
                        <span className="text-xs font-mono text-emerald-400 font-semibold">
                          Bypassed / Skipped
                        </span>
                        <p className="text-[11px] text-industrial-500">
                          Normal sample did not trigger Stage 2 defect classification.
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2 text-xs font-mono">
                        <div className="flex justify-between">
                          <span className="text-industrial-400">Predicted Class:</span>
                          <span className="text-amber-300 font-bold text-sm">
                            {inspectionResult.defect_class || 'UNKNOWN'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-industrial-400">Confidence:</span>
                          <span className="text-white font-medium">
                            {inspectionResult.classifier_confidence != null
                              ? `${(inspectionResult.classifier_confidence * 100).toFixed(1)}%`
                              : 'N/A'}
                          </span>
                        </div>

                        {/* Class Probabilities Distribution */}
                        {inspectionResult.classifier_probabilities && (
                          <div className="pt-2 border-t border-industrial-900 space-y-1">
                            <span className="text-[10px] text-industrial-400 block font-sans">
                              Class Probabilities:
                            </span>
                            {Object.entries(inspectionResult.classifier_probabilities).map(([cls, prob]) => (
                              <div key={cls} className="flex items-center justify-between text-[11px]">
                                <span className="text-industrial-400">{cls}</span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-16 bg-industrial-900 rounded-full h-1.5 overflow-hidden">
                                    <div
                                      className="bg-sky-500 h-full rounded-full"
                                      style={{ width: `${Math.min(100, Math.max(0, prob * 100))}%` }}
                                    />
                                  </div>
                                  <span className="text-white text-[10px] w-9 text-right font-mono">
                                    {(prob * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Anomaly Localization Heatmap Visualizer */}
                <div className="space-y-2">
                  <div className="flex items-center space-x-2 text-xs font-mono text-industrial-300">
                    <Layers className="w-3.5 h-3.5 text-sky-400" />
                    <span>PatchCore Anomaly Localization Heatmap</span>
                  </div>

                  {inspectionResult.heatmap_path ? (
                    <div className="rounded-lg overflow-hidden border border-industrial-800 bg-black aspect-video max-h-64 flex items-center justify-center p-1">
                      <img
                        src={getInspectionHeatmapUrl(inspectionResult.inspection_id)}
                        alt="PatchCore Anomaly Heatmap"
                        className="max-h-full max-w-full object-contain mx-auto"
                      />
                    </div>
                  ) : (
                    <div className="p-4 bg-industrial-950 rounded-lg border border-industrial-800 text-center text-xs text-industrial-500 font-mono">
                      No localization heatmap artifact generated for this run.
                    </div>
                  )}
                </div>

                {/* Traceability Metadata Footer */}
                <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3 text-[11px] font-mono text-industrial-400 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center space-x-1.5">
                      <Hash className="w-3 h-3 text-industrial-500" />
                      <span>Inspection ID:</span>
                    </span>
                    <span className="text-industrial-300 select-all">{inspectionResult.inspection_id}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center space-x-1.5">
                      <Clock className="w-3 h-3 text-industrial-500" />
                      <span>Timestamp:</span>
                    </span>
                    <span className="text-industrial-300">
                      {new Date(inspectionResult.timestamp).toLocaleString()}
                    </span>
                  </div>
                  {inspectionResult.execution_time_ms != null && (
                    <div className="flex items-center justify-between">
                      <span className="text-industrial-400">Total Inference Time:</span>
                      <span className="text-sky-300">{inspectionResult.execution_time_ms.toFixed(1)} ms</span>
                    </div>
                  )}
                </div>

                {/* Reset action */}
                <div className="flex justify-end pt-2">
                  <button
                    type="button"
                    onClick={handleResetInspection}
                    className="flex items-center space-x-2 px-4 py-2 bg-industrial-800 hover:bg-industrial-700 text-white rounded-lg text-xs font-mono transition-colors border border-industrial-700"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Inspect Next Sample</span>
                  </button>
                </div>
              </div>
            ) : (
              /* Production Standby State */
              <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-12 text-center space-y-3">
                <Cpu className="w-10 h-10 text-industrial-600 mx-auto" />
                <div className="text-sm font-semibold text-industrial-300">Inspection Station Standby</div>
                <p className="text-xs text-industrial-500 max-w-sm mx-auto">
                  Select a calibrated product on the left, start the webcam (or upload a file), and click{' '}
                  <span className="text-sky-400 font-semibold font-mono">Inspect Frame</span> to execute the two-stage inference pipeline.
                </p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
};


