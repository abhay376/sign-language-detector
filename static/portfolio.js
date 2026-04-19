// ==== Typing Effect ====
const phrases = [
    "AI-powered applications",
    "Computer Vision systems",
    "Machine Learning models",
    "full-stack web apps"
];

let phraseIndex = 0;
let charIndex = 0;
let isDeleting = false;
const typedEl = document.querySelector('.typed-text');

function type() {
    const currentPhrase = phrases[phraseIndex];

    if (isDeleting) {
        typedEl.textContent = currentPhrase.substring(0, charIndex - 1);
        charIndex--;
    } else {
        typedEl.textContent = currentPhrase.substring(0, charIndex + 1);
        charIndex++;
    }

    let speed = isDeleting ? 30 : 60;

    if (!isDeleting && charIndex === currentPhrase.length) {
        speed = 2000; // pause at end
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        phraseIndex = (phraseIndex + 1) % phrases.length;
        speed = 400;
    }

    setTimeout(type, speed);
}

type();

// ==== Navbar scroll effect ====
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// ==== Scroll reveal ====
const revealElements = document.querySelectorAll(
    '.skill-category, .project-card, .stat-card, .step, .contact-card, .about-text, .about-stats'
);

revealElements.forEach(el => el.classList.add('reveal'));

const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
            setTimeout(() => {
                entry.target.classList.add('visible');
            }, index * 80);
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

revealElements.forEach(el => observer.observe(el));

// ==== Counter animation ====
const counters = document.querySelectorAll('.stat-number');
const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const target = parseInt(entry.target.getAttribute('data-count'));
            let count = 0;
            const increment = Math.ceil(target / 30);
            const timer = setInterval(() => {
                count += increment;
                if (count >= target) {
                    entry.target.textContent = target + '+';
                    clearInterval(timer);
                } else {
                    entry.target.textContent = count;
                }
            }, 40);
            counterObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

counters.forEach(c => counterObserver.observe(c));
