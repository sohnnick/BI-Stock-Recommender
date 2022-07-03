var results_seven = JSON.parse(document.getElementById('next_seven_forecast').innerHTML);

var parent_row = document.getElementById('output-table');

for (var key in results_seven) {
	console.log(key)
  	var tr = document.createElement('tr');
  	var td1 = document.createElement('td');
  	var td2 = document.createElement('td');

	td1.innerHTML = key;
	td2.innerHTML = results_seven[key];

	tr.appendChild(td1)
	tr.appendChild(td2)

	parent_row.appendChild(tr);
};