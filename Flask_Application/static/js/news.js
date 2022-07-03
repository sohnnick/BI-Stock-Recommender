function populateNews(news_json) {
	var parent_row = document.getElementById('news-table');
	
	console.log(news_json)

	for (let i = 0; i < news_json['title'].length; i++) {
		var tr = document.createElement('tr');
		var td1 = document.createElement('td');
		var td2 = document.createElement('td');
		var td3 = document.createElement('td');
		var a = document.createElement('a');

		td1.innerHTML = news_json['publisher'][i];
		console.log(td1);

		a.innerHTML = news_json['title'][i];
		a.href = news_json['link'][i];
		a.target = "_blank";
		td2.appendChild(a);

		td3.innerHTML = news_json['providerPublishTime'][i];

		tr.appendChild(td1)
		tr.appendChild(td2)
		tr.appendChild(td3)

		parent_row.appendChild(tr);
	};
}