// API 기본 URL
const API_BASE = 'http://localhost:8000';

// DOM 요소
function getElement(id) {
    return document.getElementById(id);
}

// 토스트 메시지
function showToast(message, isSuccess = true) {
    // 간단한 알림 구현
    alert(message);
}

// 프리셋 채우기
function fillPreset(type) {
    const examples = {
        kasa: 'https://www.kasa.co.kr/faq\nhttps://kasakorea.zendesk.com/hc/ko',
        tessa: 'https://tessa.art/faq'
    };
    getElement('urlInput').value = examples[type] || '';
    getElement('sourceInput').value = type;
    showToast(`${type.toUpperCase()} 예시 불러옴`);
}

// URL 수집
async function ingestUrls() {
    const urls = getElement('urlInput').value.split('\n').filter(url => url.trim());
    if (!urls.length) {
        showToast('URL을 입력하세요', false);
        return;
    }

    const button = getElement('ingestButton');
    const spinner = getElement('ingestSpinner');
    const message = getElement('ingestMessage');
    
    button.disabled = true;
    spinner.classList.remove('is-hidden');
    message.textContent = '수집 중...';

    try {
        const response = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                urls: urls,
                source: getElement('sourceInput').value,
                use_js: getElement('jsRenderCheckbox').checked
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '수집 실패');

        message.textContent = `성공: 추가 ${data.added}개, 총 ${data.total}개`;
        showToast('수집 완료');
    } catch (error) {
        message.textContent = '오류: ' + error.message;
        showToast('수집 실패: ' + error.message, false);
    } finally {
        button.disabled = false;
        spinner.classList.add('is-hidden');
    }
}

// 질문하기
async function askQuestion() {
    const question = getElement('questionInput').value.trim();
    if (!question) {
        showToast('질문을 입력하세요', false);
        return;
    }

    const button = getElement('queryButton');
    const spinner = getElement('querySpinner');
    const results = getElement('resultsContainer');
    
    button.disabled = true;
    spinner.classList.remove('is-hidden');
    results.classList.remove('is-hidden');
    getElement('answerContent').innerHTML = '';
    getElement('sourcesContainer').innerHTML = '';

    const startTime = performance.now();

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                risk: getElement('riskSelect').value,
                budget: parseInt(getElement('budgetInput').value),
                goal: getElement('goalSelect').value,
                k: 5
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || '질문 실패');

        const endTime = performance.now();

        // 결과 표시
        getElement('backendInfo').textContent = data.backend;
        getElement('topKInfo').textContent = data.k;
        getElement('timeInfo').textContent = Math.round(endTime - startTime);
        getElement('answerContent').innerHTML = marked.parse(data.answer || '');

        // 출처 표시
        if (data.sources && data.sources.length) {
            const sourcesHTML = data.sources.map(source => `
                <li class="source-item">
                    <a href="${source.url}" target="_blank" class="source-link">
                        [${source.n}] ${source.title}
                    </a>
                    <span class="source-date">${source.as_of_date}</span>
                </li>
            `).join('');
            getElement('sourcesContainer').innerHTML = `
                <h3 class="sources-title">출처</h3>
                <ul class="sources-list">${sourcesHTML}</ul>
            `;
        }

        showToast('리포트 생성 완료');
    } catch (error) {
        showToast('생성 실패: ' + error.message, false);
        getElement('answerContent').innerHTML = `<p>오류 발생: ${error.message}</p>`;
    } finally {
        button.disabled = false;
        spinner.classList.add('is-hidden');
    }
}