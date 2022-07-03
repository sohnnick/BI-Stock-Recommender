function createCharts(results_hist, results_ana, chart_id, pred_type, price_metric) {
  // create a chart
  chart = anychart.line();

  // *******************
  // line for historical
  // *******************
  var data_graph = [];
  for(let i=0; i < results_hist['Dates'].length; i++){
    var temp_instance = new Array()
    temp_instance[0] = results_hist['Dates'][i];
    temp_instance[1] = results_hist['Open'][i]; 
    data_graph[i] = temp_instance;
  };

  var series_historical = chart.line(data_graph);
  series_historical.name("Historical");

  // *******************
  // line for analysis
  // *******************
  var data_graph = [];
  for(let i=0; i < results_ana['pred_dates'].length; i++){
    var temp_instance = new Array()
    temp_instance[0] = results_ana['pred_dates'][i];
    temp_instance[1] = results_ana['val_forecast'][i]; 
    data_graph[i] = temp_instance;
  };
  console.log(data_graph)

  var series_analysis = chart.line(data_graph);
  series_analysis.name(pred_type + " for Validation Set");

  // chart attributes
  chart.title(pred_type + ' Performance of Model on Validation Set');
  chart.xAxis().title('Date');
  chart.yAxis().title(price_metric);
  chart.legend(true);

  // set the container id
  chart.container(chart_id);

  // set colors
  chart.palette(["Green", "Yellow"]);

  // initiate drawing the chart
  chart.draw();
};