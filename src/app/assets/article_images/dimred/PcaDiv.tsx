import Plot from "react-plotly.js";

function PcaDiv() {
	const graphDiv = document.getElementById("7ff4b120-179b-4d39-964d-77f50754a7bb")
	const data = [{
		"marker": {
			"color": 2,
			"opacity": 0.8,
			"size": 10
		},
		"mode": "markers+text",
		"name": "school",
		"text": ["college", "schools", "campus", "graduate", "elementary"],
		"textfont": {
			"size": 20
		},
		"textposition": "top center",
		"x": [4.321778297424316, 3.2766661643981934, 2.796191692352295, 3.9278273582458496, 3.040142297744751],
		"y": [-1.4216220378875732, -0.5763717293739319, -0.6637948751449585, -1.2802761793136597, -0.6952879428863525],
		"z": [0.319632887840271, 0.30370813608169556, -0.7561122179031372, 0.23179687559604645, -0.4438224732875824],
		"type": "scatter3d"
	}, {
		"marker": {
			"color": 2,
			"opacity": 0.8,
			"size": 10
		},
		"mode": "markers+text",
		"name": "apple",
		"text": ["blackberry", "chips", "iphone", "microsoft", "ipad"],
		"textfont": {
			"size": 20
		},
		"textposition": "top center",
		"x": [-3.2283382415771484, -2.7246031761169434, -3.302525043487549, -1.9908421039581299, -3.0373055934906006],
		"y": [-2.0543339252471924, -0.9071000814437866, -2.518195152282715, -3.16536808013916, -2.154433012008667],
		"z": [0.27316412329673767, 2.8467764854431152, -1.7000540494918823, -0.11333499103784561, -1.6857540607452393],
		"type": "scatter3d"
	}, {
		"marker": {
			"color": 2,
			"opacity": 0.8,
			"size": 10
		},
		"mode": "markers+text",
		"name": "toilet",
		"text": ["toilets", "tub", "bathroom", "laundry", "washing"],
		"textfont": {
			"size": 20
		},
		"textposition": "top center",
		"x": [-0.5891004800796509, -1.2770955562591553, -0.5064730644226074, -0.42837440967559814, -1.1811528205871582],
		"y": [3.2948708534240723, 3.454134464263916, 3.10585618019104, 2.8472800254821777, 2.8371169567108154],
		"z": [-1.0594923496246338, -0.5471733808517456, -0.397935152053833, 0.9610295295715332, 1.1223506927490234],
		"type": "scatter3d"
	}, {
		"marker": {
			"color": "black",
			"opacity": 1,
			"size": 10
		},
		"mode": "markers+text",
		"name": "input words",
		"text": ["school", "apple", "toilet"],
		"textfont": {
			"size": 20
		},
		"textposition": "top center",
		"x": [4.053280353546143, -2.3493685722351074, -0.8007103800773621],
		"y": [-0.8719984889030457, -2.203446626663208, 2.972970724105835],
		"z": [0.07480626553297043, 1.1488221883773804, -0.5784071683883667],
		"type": "scatter3d"
	}]

	const layout = {
		"autosize": false,
		"font": {
			"family": " Courier New ",
			"size": 15
		},
		"height": 1000,
		"legend": {
			"font": {
				"color": "black",
				"family": "Courier New",
				"size": 25
			},
			"x": 1,
			"y": 0.5
		},
		"margin": {
			"b": 0,
			"l": 0,
			"r": 0,
			"t": 0
		},
		"showlegend": true,
		"width": 1000,
		"template": {
			"data": {
				"histogram2dcontour": [{
					"type": "histogram2dcontour",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					},
					"colorscale": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					]
				}],
				"choropleth": [{
					"type": "choropleth",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					}
				}],
				"histogram2d": [{
					"type": "histogram2d",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					},
					"colorscale": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					]
				}],
				"heatmap": [{
					"type": "heatmap",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					},
					"colorscale": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					]
				}],
				"heatmapgl": [{
					"type": "heatmapgl",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					},
					"colorscale": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					]
				}],
				"contourcarpet": [{
					"type": "contourcarpet",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					}
				}],
				"contour": [{
					"type": "contour",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					},
					"colorscale": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					]
				}],
				"surface": [{
					"type": "surface",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					},
					"colorscale": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					]
				}],
				"mesh3d": [{
					"type": "mesh3d",
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					}
				}],
				"scatter": [{
					"fillpattern": {
						"fillmode": "overlay",
						"size": 10,
						"solidity": 0.2
					},
					"type": "scatter"
				}],
				"parcoords": [{
					"type": "parcoords",
					"line": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"scatterpolargl": [{
					"type": "scatterpolargl",
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"bar": [{
					"error_x": {
						"color": "#2a3f5f"
					},
					"error_y": {
						"color": "#2a3f5f"
					},
					"marker": {
						"line": {
							"color": "#E5ECF6",
							"width": 0.5
						},
						"pattern": {
							"fillmode": "overlay",
							"size": 10,
							"solidity": 0.2
						}
					},
					"type": "bar"
				}],
				"scattergeo": [{
					"type": "scattergeo",
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"scatterpolar": [{
					"type": "scatterpolar",
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"histogram": [{
					"marker": {
						"pattern": {
							"fillmode": "overlay",
							"size": 10,
							"solidity": 0.2
						}
					},
					"type": "histogram"
				}],
				"scattergl": [{
					"type": "scattergl",
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"scatter3d": [{
					"type": "scatter3d",
					"line": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					},
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"scattermapbox": [{
					"type": "scattermapbox",
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"scatterternary": [{
					"type": "scatterternary",
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"scattercarpet": [{
					"type": "scattercarpet",
					"marker": {
						"colorbar": {
							"outlinewidth": 0,
							"ticks": ""
						}
					}
				}],
				"carpet": [{
					"aaxis": {
						"endlinecolor": "#2a3f5f",
						"gridcolor": "white",
						"linecolor": "white",
						"minorgridcolor": "white",
						"startlinecolor": "#2a3f5f"
					},
					"baxis": {
						"endlinecolor": "#2a3f5f",
						"gridcolor": "white",
						"linecolor": "white",
						"minorgridcolor": "white",
						"startlinecolor": "#2a3f5f"
					},
					"type": "carpet"
				}],
				"table": [{
					"cells": {
						"fill": {
							"color": "#EBF0F8"
						},
						"line": {
							"color": "white"
						}
					},
					"header": {
						"fill": {
							"color": "#C8D4E3"
						},
						"line": {
							"color": "white"
						}
					},
					"type": "table"
				}],
				"barpolar": [{
					"marker": {
						"line": {
							"color": "#E5ECF6",
							"width": 0.5
						},
						"pattern": {
							"fillmode": "overlay",
							"size": 10,
							"solidity": 0.2
						}
					},
					"type": "barpolar"
				}],
				"pie": [{
					"automargin": true,
					"type": "pie"
				}]
			},
			"layout": {
				"autotypenumbers": "strict",
				"colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"],
				"font": {
					"color": "#2a3f5f"
				},
				"hovermode": "closest",
				"hoverlabel": {
					"align": "left"
				},
				"paper_bgcolor": "white",
				"plot_bgcolor": "#E5ECF6",
				"polar": {
					"bgcolor": "#E5ECF6",
					"angularaxis": {
						"gridcolor": "white",
						"linecolor": "white",
						"ticks": ""
					},
					"radialaxis": {
						"gridcolor": "white",
						"linecolor": "white",
						"ticks": ""
					}
				},
				"ternary": {
					"bgcolor": "#E5ECF6",
					"aaxis": {
						"gridcolor": "white",
						"linecolor": "white",
						"ticks": ""
					},
					"baxis": {
						"gridcolor": "white",
						"linecolor": "white",
						"ticks": ""
					},
					"caxis": {
						"gridcolor": "white",
						"linecolor": "white",
						"ticks": ""
					}
				},
				"coloraxis": {
					"colorbar": {
						"outlinewidth": 0,
						"ticks": ""
					}
				},
				"colorscale": {
					"sequential": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					],
					"sequentialminus": [
						[0.0, "#0d0887"],
						[0.1111111111111111, "#46039f"],
						[0.2222222222222222, "#7201a8"],
						[0.3333333333333333, "#9c179e"],
						[0.4444444444444444, "#bd3786"],
						[0.5555555555555556, "#d8576b"],
						[0.6666666666666666, "#ed7953"],
						[0.7777777777777778, "#fb9f3a"],
						[0.8888888888888888, "#fdca26"],
						[1.0, "#f0f921"]
					],
					"diverging": [
						[0, "#8e0152"],
						[0.1, "#c51b7d"],
						[0.2, "#de77ae"],
						[0.3, "#f1b6da"],
						[0.4, "#fde0ef"],
						[0.5, "#f7f7f7"],
						[0.6, "#e6f5d0"],
						[0.7, "#b8e186"],
						[0.8, "#7fbc41"],
						[0.9, "#4d9221"],
						[1, "#276419"]
					]
				},
				"xaxis": {
					"gridcolor": "white",
					"linecolor": "white",
					"ticks": "",
					"title": {
						"standoff": 15
					},
					"zerolinecolor": "white",
					"automargin": true,
					"zerolinewidth": 2
				},
				"yaxis": {
					"gridcolor": "white",
					"linecolor": "white",
					"ticks": "",
					"title": {
						"standoff": 15
					},
					"zerolinecolor": "white",
					"automargin": true,
					"zerolinewidth": 2
				},
				"scene": {
					"xaxis": {
						"backgroundcolor": "#E5ECF6",
						"gridcolor": "white",
						"linecolor": "white",
						"showbackground": true,
						"ticks": "",
						"zerolinecolor": "white",
						"gridwidth": 2
					},
					"yaxis": {
						"backgroundcolor": "#E5ECF6",
						"gridcolor": "white",
						"linecolor": "white",
						"showbackground": true,
						"ticks": "",
						"zerolinecolor": "white",
						"gridwidth": 2
					},
					"zaxis": {
						"backgroundcolor": "#E5ECF6",
						"gridcolor": "white",
						"linecolor": "white",
						"showbackground": true,
						"ticks": "",
						"zerolinecolor": "white",
						"gridwidth": 2
					}
				},
				"shapedefaults": {
					"line": {
						"color": "#2a3f5f"
					}
				},
				"annotationdefaults": {
					"arrowcolor": "#2a3f5f",
					"arrowhead": 0,
					"arrowwidth": 1
				},
				"geo": {
					"bgcolor": "white",
					"landcolor": "#E5ECF6",
					"subunitcolor": "white",
					"showland": true,
					"showlakes": true,
					"lakecolor": "white"
				},
				"title": {
					"x": 0.05
				},
				"mapbox": {
					"style": "light"
				}
			}
		}
	}

	const config = 
		function(Plotly: { purge: (arg0: HTMLElement | null) => void; }) {

		var gd = document.getElementById('7ff4b120-179b-4d39-964d-77f50754a7bb');
		var x = new MutationObserver(function(mutations, observer) {
			var display = null
			if (gd) {
				display = window.getComputedStyle(gd).display;
			}
			if (!display || display === 'none') {
				console.log([gd, 'removed!']);
				Plotly.purge(gd);
				observer.disconnect();
			}
		});

		// Listen for the removal of the full notebook cells
		var notebookContainer = gd?.closest('#notebook-container');
		if (notebookContainer) {
			x.observe(notebookContainer, {
				childList: true
			});
		}

		// Listen for the clearing of the current output cell
		var outputEl = gd?.closest('.output');
		if (outputEl) {
			x.observe(outputEl, {
				childList: true
			});
		}
	}

		return (
			<div id="7ff4b120-179b-4d39-964d-77f50754a7bb" className="plotly-graph-div" style={{height: "1000px", width: "1000px"}}>
				<Plot
					graphDiv={graphDiv}
					data={data}
					layout={layout}
					config={config}
				/>
			</div>
		)
	}

export default PcaDiv;