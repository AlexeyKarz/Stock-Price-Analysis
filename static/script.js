function findSymbol() {
    var companyName = document.getElementById('company-name').value;
    if (companyName.length > 1) {
        $.ajax({
            url: "/search_stock_symbols",
            dataType: "json",
            data: { query: companyName },
            success: function(data) {
                var resultsContainer = document.getElementById('search-results');
                resultsContainer.innerHTML = ''; // Clear previous results
                if (data.length) {
                    var list = '<ul>';
                    data.forEach(function(item) {
                        list += `<li><div class="company-info">${item.name} (${item.symbol})</div><div class="use-button"><button type='button' onclick="setSymbol('${item.symbol}')">Use</button></div></li>`;
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

// function displayStockOverview(symbol) {
//     fetch(`/stock_overview/${symbol}`)  // Adjusted to the correct endpoint
//         .then(response => response.json())
//         .then(data => {
//             if (data.error) {
//                 document.getElementById('stock-overview').innerHTML = data.error;
//             } else {
//                 // Assuming all keys are properly named and the data structure is as expected
//                 const info = `
//                     <h3>Stock Overview for ${data.Symbol}</h3>
//                     <p><strong>Company:</strong> ${data.Name}</p>
//                     <p><strong>Sector:</strong> ${data.Sector}</p>
//                     <p><strong>Industry:</strong> ${data.Industry}</p>
//                     <p><strong>Description:</strong> ${data.Description}</p>
//                     <p><strong>Market Cap:</strong> ${data.MarketCapitalization}</p>
//                     <p><strong>EBITDA:</strong> ${data.EBITDA}</p>
//                     <p><strong>PE Ratio:</strong> ${data.PERatio}</p>
//                     <p><strong>Dividend Yield:</strong> ${data.DividendYield}</p>
//                     <!-- Add more fields as necessary -->
//                 `;
//                 document.getElementById('stock-overview').innerHTML = info;
//             }
//         })
//         .catch(error => console.error('Error fetching stock overview:', error));
// }
//
