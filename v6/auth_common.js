// ===== 전역 인증 관리 스크립트 =====
// 모든 페이지에서 사용할 공통 인증 함수

function updateAuthUI() {
    const token = localStorage.getItem('access_token');
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    
    const authSection = document.getElementById('auth-section');
    const profileLink = document.getElementById('profile-link');
    
    if (!authSection) {
        console.warn('auth-section 요소를 찾을 수 없습니다.');
        return;
    }
    
    if (token && user.email) {
        authSection.innerHTML = `
            <span id="user-name" style="margin-right: 1rem; color: #6B7280;">${user.name}님</span>
            <button onclick="logout()" class="nav-button" style="background: #EF4444;">로그아웃</button>
        `;
        if (profileLink) profileLink.style.display = 'block';
    } else {
        authSection.innerHTML = `
            <a href="/login.html" class="nav-button" id="login-button">로그인</a>
        `;
        if (profileLink) profileLink.style.display = 'none';
    }
}

function logout() {
    if (confirm('로그아웃 하시겠습니까?')) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('user');
        window.location.href = '/';
    }
}

// 페이지 로드시 인증 상태 확인
document.addEventListener('DOMContentLoaded', function() {
    updateAuthUI();
});