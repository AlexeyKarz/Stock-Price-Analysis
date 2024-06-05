function findSymbol() {
    var companyName = document.getElementById('company-name').value;
    if (companyName.length > 1) {  // Check if the input is not too short
        $.ajax({
            url: "/search_stock_symbols",
            dataType: "json",
            data: {
                query: companyName
            },
            success: function(data) {
                var resultsContainer = document.getElementById('search-results');
                resultsContainer.innerHTML = ''; // Clear previous results
                if (data.length) {
                    var list = '<ul>';
                    data.forEach(function(item) {
                        list += `<li>${item.name} (${item.symbol}) - <button onclick="setSymbol('${item.symbol}')">Use</button></li>`;
                    });
                    list += '</ul>';
                    resultsContainer.innerHTML = list;
                } else {
                    resultsContainer.innerHTML = 'No results found';
                }
            }
        });
    }
}

function setSymbol(symbol) {
    document.getElementById('stock-name').value = symbol;
}
