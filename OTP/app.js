const otpContainer = document.querySelector('.otp-container');
const mobileVerify = document.querySelector('.mobile-verify');
const boxVerify = document.querySelector('.box-verify');
const btnContinue = document.querySelector('.btn-continue');
const btnResend = document.querySelector('.btn-resend');
const btnVerify = document.querySelector('.btn-verify');
const btnBack = document.querySelector('.btn-back');
const phoneNumber = document.getElementById('phone-number');
const otpInput = document.querySelectorAll('.otp-input .input');
const containerContent = document.querySelector('.container-content');
const expireEle = document.querySelector('.expire');


let expire = 30;
let OTP;
let countdown;
let yourInputNumber = '';

btnContinue.addEventListener('click', () => {
    
    resetStateOTP();
    
    const phoneNumberExist = phoneNumber.value.match(/^\d{10}$/g);

    if (phoneNumberExist) {
        
        otpContainer.classList.remove('go-right');
        mobileVerify.classList.remove('go-right');
        
        otpContainer.classList.add('active-box');
        mobileVerify.classList.remove('active-box');

        
        alertText('.phone-num-input .text-danger', '');
        
        document.querySelector('.phone').textContent = formatPhoneNumber(
            phoneNumberExist
        );
        
        OTP = randomOTP();
        handleCountDown();
        alert(`Your OTP: ${OTP}`);
        console.log(OTP);
    } else {
        alertText(
            '.phone-num-input .text-danger',
            'Please enter a valid phone number'
        );
    }
});


btnBack.addEventListener('click', () => {
    otpContainer.classList.add('go-right');
    otpContainer.classList.remove('active-box');
    mobileVerify.classList.add('go-right', 'active-box');
});


otpInput.forEach((input) => {
    input.addEventListener('keyup', (e) => {
        const element = e.target;

        if (element.value.match(/\d/)) {
            yourInputNumber += element.value;
            alertText('.otp-container .text-danger', '');

            if (element.nextElementSibling) {
                element.nextElementSibling.focus();
            }
        } else {
            alertText(
                '.otp-container .text-danger',
                'Enter a number in each field'
            );
        }
    });
});


btnVerify.addEventListener('click', () => {
    const icon = boxVerify.querySelector('.fas');

    if (OTP === yourInputNumber) {
        icon.classList.add('fa-check-circle');
        icon.classList.remove('fa-times-circle');
        boxVerify.querySelector('p').innerHTML = `
        Your account has been <br/> verified successfully
        <br/>
        <span class='text-muted'>Please wait while redirecting</span>
        `;

        setTimeout(() => {
            window.location.href = `http://127.0.0.1:5000`;
        }, 3000);
    } else {
        icon.classList.remove('fa-check-circle');
        icon.classList.add('fa-times-circle');
        boxVerify.querySelector('p').innerHTML = `
        Verification failed
        <br/>
        <span class='text-muted'>Please <span class='btn-return text-dark'>try again</span></span>
        `;
    }
    boxVerify.classList.add('active');
});


containerContent.addEventListener('click', (e) => {
    const element = e.target;
    if (element.classList.contains('btn-return')) {
        boxVerify.classList.remove('active');

        activeStateOTP();
    }
});


btnResend.addEventListener('click', activeStateOTP);


window.addEventListener('keydown', (e) => {
    const key = e.key;
    const keyEle = document.querySelector(
        `.active-box span[data-key='${key.toLowerCase()}']`
    );

    if (keyEle) {
        keyEle.classList.add('active');
        setTimeout(() => {
            keyEle.classList.remove('active');
        }, 500);
    }
});

function handleCountDown() {
    countdown = setInterval(() => {
        expire--;
        if (expire === 0) {
            clearInterval(countdown);
            OTP = null;
            console.log(OTP);
        }
        expireEle.textContent = expire < 10 ? '0' + expire + 's' : expire + 's';
    }, 1000);
}

function alertText(element, text) {
    document.querySelector(`${element}`).textContent = text;
}

function randomOTP() {
    let random = '';
    Array.from({ length: 4 }, () => {
        random += Math.floor(Math.random() * 10).toString();
    });
    return random;
}
function resetStateOTP() {
    clearInterval(countdown);
    expire = 30;
    OTP = null;
    yourInputNumber = '';

    otpInput.forEach((input) => {
        input.value = '';
    });
}
function formatPhoneNumber(number) {
    return number.toString().slice(0, 7) + '***';
}

function activeStateOTP() {
    resetStateOTP();

    OTP = randomOTP();
    handleCountDown();
    alert(`Your OTP: ${OTP}`);
    console.log(OTP);
}
