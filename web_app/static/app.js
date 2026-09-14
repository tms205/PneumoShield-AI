// ==========================================
// CẤU HÌNH THÔNG TIN MÔ HÌNH (METRICS)
// ==========================================
const MODEL_METRICS_INFO = {
    efficientnet: {
        accuracy: "94.89%",
        auc: "0.9829",
        imgsize: "380 x 380 pixels",
        description: "Thông số lấy từ outputs_scratch/efficientnet_b4/test_evaluation.png trên tập test: confusion matrix [[1277, 52], [62, 841]], recall viêm phổi 93.13%, precision viêm phổi 94.18%."
    },
    vit: {
        accuracy: "86.07%",
        auc: "0.9481",
        imgsize: "224 x 224 pixels",
        description: "Thông số lấy từ outputs_scratch/vit_b_16/test_evaluation.png trên tập test: confusion matrix [[1097, 232], [79, 824]], recall viêm phổi 91.25%, precision viêm phổi 78.03%."
    },
    ensemble: {
        accuracy: "91.36%",
        auc: "0.9690",
        imgsize: "Mỗi mô hình tự động resize riêng",
        description: "Thông số ensemble tính theo công thức bạn chọn: 0.6 * EfficientNet-B4 + 0.4 * ViT-B/16. Đây là chỉ số phối tuyến tính từ hai model riêng lẻ, chưa phải kết quả đánh giá lại trực tiếp trên toàn bộ tập test."
    }
};

// ==========================================
// TRẠNG THÁI ỨNG DỤNG (STATE)
// ==========================================
let activeMode = 'single'; // 'single' hoặc 'ensemble'
let selectedSingleModel = 'efficientnet'; // 'efficientnet' hoặc 'vit'
let effWeight = 0.5;
let vitWeight = 0.5;

let selectedFile = null;      // Chứa đối tượng File tự upload
let selectedSample = null;    // Chứa tên file ảnh mẫu
let previewUrl = null;        // URL object preview ảnh tự upload

// Khởi chạy khi tài liệu HTML sẵn sàng
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

// ==========================================
// KHỞI TẠO & LIÊN KẾT SỰ KIỆN
// ==========================================
function initApp() {
    // 1. Cập nhật giao diện thông số ban đầu
    updateMetricsPanel();
    
    // 2. Tải danh sách ảnh mẫu từ Server
    fetchSamples();

    // 3. Tải ảnh kết quả huấn luyện từ outputs_scratch
    fetchTrainingOutputs();
    
    // 4. Thiết lập các sự kiện kéo thả cho vùng Upload
    setupDragAndDrop();
}

// Cập nhật thẻ hiển thị thông số mô hình
function updateMetricsPanel() {
    let metricsKey = activeMode === 'ensemble' ? 'ensemble' : selectedSingleModel;
    let data = MODEL_METRICS_INFO[metricsKey];
    const customEnsembleWeights = activeMode === 'ensemble' && (effWeight !== 0.5 || vitWeight !== 0.5);
    
    document.getElementById("metric-accuracy").innerText = customEnsembleWeights ? "Chưa đánh giá" : data.accuracy;
    document.getElementById("metric-auc").innerText = customEnsembleWeights ? "Chưa đánh giá" : data.auc;
    document.getElementById("metric-imgsize").innerText = data.imgsize;
    document.getElementById("metric-description").innerText = customEnsembleWeights
        ? "Bạn đang sử dụng trọng số tùy chỉnh. Ứng dụng có thể suy luận, nhưng chưa có chỉ số test được đo cho cấu hình này."
        : data.description;
}

// ==========================================
// THIẾT LẬP CHẾ ĐỘ CHẠY (SINGLE / ENSEMBLE)
// ==========================================
function setMode(mode) {
    activeMode = mode;
    
    // Đổi trạng thái hiển thị của tab nút
    if (mode === 'single') {
        document.getElementById("tab-single").classList.add("active");
        document.getElementById("tab-ensemble").classList.remove("active");
        
        document.getElementById("single-config-panel").classList.remove("hidden");
        document.getElementById("ensemble-config-panel").classList.add("hidden");
    } else {
        document.getElementById("tab-single").classList.remove("active");
        document.getElementById("tab-ensemble").classList.add("active");
        
        document.getElementById("single-config-panel").classList.add("hidden");
        document.getElementById("ensemble-config-panel").classList.remove("hidden");
    }
    
    // Cập nhật lại thẻ thông số & tính toán lại nếu đang hiển thị kết quả
    updateMetricsPanel();
    
    // Nếu kết quả chẩn đoán trước đó đang hiển thị, chạy lại chẩn đoán để cập nhật theo cấu hình mới
    if (selectedFile || selectedSample) {
        // Tự động chẩn đoán lại khi đổi cài đặt mô hình
        runDiagnosis();
    }
}

// Cập nhật khi chọn radio trong đơn mô hình
function updateSingleModelView() {
    const checkedRadio = document.querySelector('input[name="single-model-select"]:checked');
    selectedSingleModel = checkedRadio.value;
    updateMetricsPanel();
    
    if (selectedFile || selectedSample) {
        runDiagnosis();
    }
}

// Đồng bộ hóa giá trị hai thanh trượt trọng số (Tổng luôn = 100%)
function syncWeights(triggerSource) {
    const sliderEff = document.getElementById("slider-eff");
    const sliderVit = document.getElementById("slider-vit");
    const valEffText = document.getElementById("val-eff-weight");
    const valVitText = document.getElementById("val-vit-weight");
    
    let effVal = parseInt(sliderEff.value);
    let vitVal = parseInt(sliderVit.value);
    
    if (triggerSource === 'eff') {
        vitVal = 100 - effVal;
        sliderVit.value = vitVal;
    } else {
        effVal = 100 - vitVal;
        sliderEff.value = effVal;
    }
    
    // Cập nhật text hiển thị tỷ lệ %
    valEffText.innerText = `${effVal}%`;
    valVitText.innerText = `${vitVal}%`;
    
    // Chuyển đổi thành giá trị số thập phân [0.0 - 1.0]
    effWeight = effVal / 100.0;
    vitWeight = vitVal / 100.0;
    updateMetricsPanel();
    
    // Tự động tính toán chẩn đoán lại sau một khoảng delay nhỏ tránh spam request liên tục
    debounce(runDiagnosis, 300)();
}

// Hàm trì hoãn (Debounce) giúp giảm số lần gọi API khi kéo slider nhanh
let debounceTimer;
function debounce(func, delay) {
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => func.apply(context, args), delay);
    };
}

// ==========================================
// THƯ VIỆN ẢNH MẪU (SAMPLES)
// ==========================================
async function fetchSamples() {
    const gallery = document.getElementById("samples-gallery");
    try {
        const response = await fetch("/api/samples");
        if (!response.ok) throw new Error("Không thể lấy danh sách ảnh mẫu.");
        
        const samples = await response.ok ? await response.json() : [];
        
        if (samples.length === 0) {
            gallery.innerHTML = '<div class="gallery-placeholder">Không tìm thấy ảnh mẫu trên máy chủ.</div>';
            return;
        }
        
        gallery.innerHTML = ""; // Xóa placeholder
        
        samples.forEach(sample => {
            const item = document.createElement("div");
            item.className = "gallery-item animate-fade-in";
            item.onclick = () => selectSample(sample.filename);
            
            const badgeClass = sample.label === 'NORMAL' ? 'badge-normal' : 'badge-pneumonia';
            const vietnameseLabel = sample.label === 'NORMAL' ? 'Bình thường' : 'Viêm phổi';
            
            item.innerHTML = `
                <div class="gallery-thumb-container">
                    <img src="${sample.url}" alt="${sample.filename}">
                    <span class="gallery-badge ${badgeClass}">${sample.label}</span>
                </div>
                <span class="gallery-item-lbl" title="${sample.filename}">${sample.filename.substring(sample.filename.indexOf('_') + 1)}</span>
            `;
            gallery.appendChild(item);
        });
        
    } catch (error) {
        console.error(error);
        gallery.innerHTML = `<div class="gallery-placeholder text-danger"><i class="fa-solid fa-triangle-exclamation"></i> Lỗi kết nối máy chủ ảnh mẫu.</div>`;
    }
}

// ==========================================
// KET QUA HUAN LUYEN (SCRATCH)
// ==========================================
async function fetchTrainingOutputs() {
    const container = document.getElementById("training-outputs");
    if (!container) return;

    try {
        const response = await fetch("/api/training-outputs");
        if (!response.ok) throw new Error("Khong the lay danh sach output.");

        const runs = await response.json();
        const availableRuns = runs.filter(run => run.exists && run.models.length > 0);

        if (availableRuns.length === 0) {
            container.innerHTML = '<div class="gallery-placeholder">Khong tim thay anh output trong outputs_scratch.</div>';
            return;
        }

        container.innerHTML = availableRuns.map(run => `
            <section class="output-run">
                <div class="output-run-header">
                    <h3>${escapeHtml(run.label)}</h3>
                    <span>${run.models.length} models</span>
                </div>
                ${run.models.map(model => `
                    <div class="output-model">
                        <div class="output-model-title">${escapeHtml(model.label)}</div>
                        <div class="output-image-grid">
                            ${model.images.map(image => `
                                <a class="output-image-card" href="${image.url}" target="_blank" rel="noopener">
                                    <img src="${image.url}" alt="${escapeHtml(image.title)}">
                                    <span>${escapeHtml(image.title)}</span>
                                </a>
                            `).join("")}
                        </div>
                    </div>
                `).join("")}
            </section>
        `).join("");
    } catch (error) {
        console.error(error);
        container.innerHTML = '<div class="gallery-placeholder text-danger"><i class="fa-solid fa-triangle-exclamation"></i> Loi ket noi output huan luyen.</div>';
    }
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[char]);
}

// Khi người dùng bấm chọn một ảnh mẫu
function selectSample(filename) {
    resetUploadStateOnly();
    selectedSample = filename;
    
    // Hiển thị ảnh xem trước
    const previewContainer = document.getElementById("preview-container");
    const imgPreview = document.getElementById("image-preview");
    const uploadCard = document.getElementById("upload-card");
    
    imgPreview.src = `/static/samples/${filename}`;
    previewContainer.classList.remove("hidden");
    uploadCard.classList.add("drop-zone-active");
    
    // Tự động chẩn đoán ngay
    runDiagnosis();
}

// ==========================================
// TỰ TẢI ẢNH QUA FILE CHỌN HOẶC KÉO THẢ
// ==========================================
function triggerFileInput() {
    document.getElementById("file-input").click();
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        processUploadedFile(files[0]);
    }
}

function setupDragAndDrop() {
    const dropZone = document.getElementById("upload-card");
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processUploadedFile(files[0]);
        }
    }, false);
}

function processUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Vui lòng chỉ tải lên tệp tin định dạng hình ảnh.');
        return;
    }
    
    resetUploadStateOnly();
    selectedFile = file;
    
    // Tạo preview ảnh
    previewUrl = URL.createObjectURL(file);
    const previewContainer = document.getElementById("preview-container");
    const imgPreview = document.getElementById("image-preview");
    
    imgPreview.src = previewUrl;
    previewContainer.classList.remove("hidden");
}

// Xóa trạng thái tải hình cũ
function resetUploadStateOnly() {
    selectedFile = null;
    selectedSample = null;
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
    document.getElementById("preview-container").classList.add("hidden");
    document.getElementById("image-preview").src = "";
    
    // Ẩn vùng kết quả và hiện placeholder ban đầu
    document.getElementById("results-content").classList.add("hidden");
    document.getElementById("loading-state").classList.add("hidden");
    document.getElementById("result-placeholder").classList.remove("hidden");
}

function resetUpload() {
    resetUploadStateOnly();
    document.getElementById("file-input").value = "";
}

// ==========================================
// THỰC HIỆN GỌI API CHẨN ĐOÁN
// ==========================================
async function runDiagnosis() {
    if (!selectedFile && !selectedSample) {
        return;
    }
    
    // 1. Cập nhật giao diện trạng thái Đang tải (Loading)
    document.getElementById("result-placeholder").classList.add("hidden");
    document.getElementById("results-content").classList.add("hidden");
    document.getElementById("loading-state").classList.remove("hidden");
    
    // Cuộn màn hình nhẹ đến vùng kết quả trên mobile
    document.getElementById("results-card").scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // 2. Tính toán trọng số gửi đi dựa trên chế độ cấu hình
    let currentWEff = 0.5;
    let currentWVit = 0.5;
    
    if (activeMode === 'single') {
        currentWEff = selectedSingleModel === 'efficientnet' ? 1.0 : 0.0;
        currentWVit = selectedSingleModel === 'vit' ? 1.0 : 0.0;
    } else {
        currentWEff = effWeight;
        currentWVit = vitWeight;
    }
    
    // 3. Xây dựng URL và body gọi API
    let url = `/api/predict?w_eff=${currentWEff}&w_vit=${currentWVit}`;
    let fetchOptions = {};
    
    if (selectedSample) {
        // Dự đoán ảnh mẫu: chỉ gửi tên file qua query parameter
        url += `&sample=${selectedSample}`;
        fetchOptions = {
            method: 'POST'
        };
    } else if (selectedFile) {
        // Upload ảnh tự chọn: gửi raw dữ liệu nhị phân qua Multipart Form Data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        fetchOptions = {
            method: 'POST',
            body: formData
        };
    }
    
    // 4. Thực thi Request
    try {
        const response = await fetch(url, fetchOptions);
        if (!response.ok) throw new Error("Chẩn đoán thất bại từ máy chủ.");
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert(`Lỗi chẩn đoán: ${data.error}`);
            resetUpload();
        }
    } catch (error) {
        console.error(error);
        alert(`Không thể kết nối máy chủ AI: ${error.message}`);
        resetUpload();
    }
}

// ==========================================
// HIỂN THỊ KẾT QUẢ PHÂN TÍCH (VISUALIZATION)
// ==========================================
function displayResults(data) {
    // Ẩn loading và hiện khối kết quả
    document.getElementById("loading-state").classList.add("hidden");
    const resultsContent = document.getElementById("results-content");
    resultsContent.classList.remove("hidden");
    
    // 1. Nhãn kết quả chính (NORMAL / PNEUMONIA)
    const predBadge = document.getElementById("pred-badge");
    const predConfidenceText = document.getElementById("pred-confidence");
    
    const isPneumonia = data.prediction === 'PNEUMONIA';
    predBadge.innerText = isPneumonia ? 'PNEUMONIA (VIÊM PHỔI)' : 'NORMAL (BÌNH THƯỜNG)';
    
    // Đổi màu badge nhãn chính
    if (isPneumonia) {
        predBadge.className = "badge pneumonia";
    } else {
        predBadge.className = "badge normal";
    }
    
    // Hiển thị % tự tin
    const confidencePercent = (data.confidence * 100).toFixed(1);
    predConfidenceText.innerText = `Độ tin cậy: ${confidencePercent}%`;
    
    // 2. Cập nhật thanh đo xác suất tổng hợp của viêm phổi
    const probBar = document.getElementById("prob-bar");
    const probPercentageText = document.getElementById("prob-percentage");
    
    // Xác suất viêm phổi tổng hợp (Ensemble)
    const pPneumonia = data.ensemble.pneumonia;
    const pPneumoniaPercent = (pPneumonia * 100).toFixed(1);
    
    probPercentageText.innerText = `${pPneumoniaPercent}%`;
    
    // Thiết lập độ rộng của thanh bar
    setTimeout(() => {
        probBar.style.width = `${pPneumoniaPercent}%`;
    }, 50);
    
    // Thay đổi màu sắc thanh bar tổng hợp tùy theo chẩn đoán chính
    if (isPneumonia) {
        probBar.className = "prob-bar"; // Đỏ-Cam mặc định
    } else {
        probBar.className = "prob-bar is-normal"; // Xanh lá
    }
    
    // 3. Thời gian xử lý & cấu hình mô hình
    document.getElementById("val-time-taken").innerText = `${data.time_taken_sec.toFixed(2)}s`;
    
    let configStr = "Ensemble";
    if (activeMode === 'single') {
        configStr = selectedSingleModel === 'efficientnet' ? "Đơn mô hình (EfficientNet)" : "Đơn mô hình (ViT)";
    }
    document.getElementById("val-active-config").innerText = configStr;
    
    // 4. Phân tích chi tiết từng mô hình con (Breakdown)
    const pEffPneumonia = data.efficientnet.pneumonia;
    const pVitPneumonia = data.vit.pneumonia;
    
    const pctEff = (pEffPneumonia * 100).toFixed(1);
    const pctVit = (pVitPneumonia * 100).toFixed(1);
    
    document.getElementById("breakdown-eff-pct").innerText = `${pctEff}%`;
    document.getElementById("breakdown-vit-pct").innerText = `${pctVit}%`;
    
    // Trọng số mô hình
    document.getElementById("breakdown-eff-w").innerText = `${(data.ensemble.w_eff * 100).toFixed(0)}%`;
    document.getElementById("breakdown-vit-w").innerText = `${(data.ensemble.w_vit * 100).toFixed(0)}%`;
    
    // Thời gian suy luận mô hình con
    document.getElementById("breakdown-eff-time").innerText = `${data.efficientnet.time_sec.toFixed(3)}s`;
    document.getElementById("breakdown-vit-time").innerText = `${data.vit.time_sec.toFixed(3)}s`;
    
    // Thiết lập độ rộng thanh mini-bar chi tiết
    setTimeout(() => {
        document.getElementById("breakdown-eff-bar").style.width = `${pctEff}%`;
        document.getElementById("breakdown-vit-bar").style.width = `${pctVit}%`;
    }, 150);
    
    // 5. Cập nhật ô hiển thị công thức kết hợp
    const formulaText = document.getElementById("ensemble-formula");
    
    let w1 = data.ensemble.w_eff.toFixed(2);
    let w2 = data.ensemble.w_vit.toFixed(2);
    
    let pEffNormal = data.efficientnet.normal.toFixed(4);
    let pEffPneu = data.efficientnet.pneumonia.toFixed(4);
    
    let pVitNormal = data.vit.normal.toFixed(4);
    let pVitPneu = data.vit.pneumonia.toFixed(4);
    
    let pEnsNormal = data.ensemble.normal.toFixed(4);
    let pEnsPneu = data.ensemble.pneumonia.toFixed(4);
    
    formulaText.innerHTML = `
P_Ensemble(Normal) = (${w1} * ${pEffNormal}) + (${w2} * ${pVitNormal}) = ${pEnsNormal}
P_Ensemble(Pneumonia) = (${w1} * ${pEffPneu}) + (${w2} * ${pVitPneu}) = ${pEnsPneu}
👉 Nhãn dự đoán: ${data.prediction} (${confidencePercent}%)
    `.trim();
}
