document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results');
    const loadingIndicator = document.getElementById('loading');
    const precioEUR = document.getElementById('precio-eur');
    const precioUSD = document.getElementById('precio-usd');

    const API_URL = 'http://localhost:5000';

    const formGroups = document.querySelectorAll('.form-group');
    formGroups.forEach((group, index) => {
        group.style.animationDelay = `${0.1 * (index + 1)}s`;
    });
    
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });

    function enviarDatos(formData) {
        return fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor');
            }
            return response.json();
        });
    }

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        loadingIndicator.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        
        const formData = {
            brand: document.getElementById('brand').value,
            model: document.getElementById('model').value,
            vehicle_type: document.getElementById('vehicle_type').value,
            gearbox: document.getElementById('gearbox').value,
            fuel_type: document.getElementById('fuel_type').value,
            power: parseInt(document.getElementById('power').value),
            mileage: parseInt(document.getElementById('mileage').value),
            car_age: parseInt(document.getElementById('car_age').value),
            not_repaired: parseInt(document.getElementById('not_repaired').value)
        };
        
        enviarDatos(formData)
        .then(data => {
            loadingIndicator.classList.add('hidden');
            
            precioEUR.textContent = `${data.eur.toLocaleString('es-ES')} €`;
            precioUSD.textContent = `$${data.usd.toLocaleString('en-US')}`;
            
            resultsContainer.classList.remove('hidden');
            
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.classList.add('hidden');
            alert('Error al procesar la solicitud. Por favor, inténtalo de nuevo.');
        });
    });

    document.getElementById('reset-button').addEventListener('click', function() {
        resultsContainer.classList.add('hidden');
    });
});
