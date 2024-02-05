def linearthreshold_model(graph, r, initial_p0=0.5, min_wakeup_condition=None, 
                          min_wakeup_per_node_condition=None):
    """
    Simulate an linear threshold model dynamic graph
    Args:
        graph : input nodes and edges ;
        r : linear threshold model parameter ;
        initial_p0 : initial probability P(node_state=0).
        min_wakeup_condition (int): simulation stops when this number of events are concluded ;
        min_wakeup_per_node_condition (int): simulation stops when all nodes wake up at least
            this amount of times ;
    Returns:
        graph: final state of the graph
    """    
    graph = graph.copy()
    n = graph.number_of_nodes()
    
    wakeup_total_count = 0
    min_wakeup_per_node = 0
    nodes_at_min_wakeup = 0
    # consider property of poisson processes - any node wake up with rate lambda_n
    lambda_n = LAMBDA_V * n

    # initialization
    for v in graph.nodes:
        graph.nodes[v]['state'] = 0 if np.random.uniform(0,1)<initial_p0 else +1
        graph.nodes[v]['wakeup_count'] = 0
    
    # first event
    node = np.random.choice( graph.nodes )
    event = Event( time=0, node=node )
    next_event = event
    time=0
    events=0

    print('LINEAR THRESHOLD')
    stop_sim = False
    while not stop_sim:
        wakeup_total_count += 1
        event = next_event
        events+=1
        time, v = event.time, event.node
        # get current state from node neighbors
        neighbors_state = np.fromiter((graph.nodes[nbor]['state'] for nbor in graph.neighbors(v)), dtype='?')
        Nv = np.sum(neighbors_state)
        graph.nodes[v]['state'] = 1 if Nv>r else 0

        next_wakeup_time = time + np.random.exponential( scale=1/lambda_n )
        next_wakeup_node = np.random.choice(graph.nodes)
        next_event = Event(time=next_wakeup_time, node=next_wakeup_node) )

        print(f'\rNodes wake up: {wakeup_count}/{min_wakeup_condition} - Time: {time:.5f}', end='', flush=True)
        # print(f'\rTime: {time:.5f}', end='', flush=True)

    print(f'\nEvents: {events}')


# QQ plot
def degrees_distribution_analysis(graph, graph_n, graph_p, conf_level=0.95):
    # graph_n and graph_p were used to generate the random graph
    poisson_np = graph_n*graph_p
    degrees = [ node.degree for node in graph.nodes ]
    # number of non isolated nodes on the graph
    n = graph.number_of_nodes
    
    quantiles_empiric = np.sort(degrees)    
    quantiles_theory = np.zeros(n)
    # skip first since for large n, the first quantile is already zero
    for i in range(1,n):
        quantiles_theory[i] = stats.poisson.ppf(i/n, poisson_np)

    # Chi-Square Goodness of Fit Test 
    # get frequency of observed and expected values
    empiric_values, empiric_count = np.unique(quantiles_empiric, return_counts=True)
    theory_values, theory_count = np.unique(quantiles_theory, return_counts=True)
    empiric_count = { empiric_values[j]:empiric_count[j] for j in range(len(empiric_values)) }
    theory_count = { theory_values[j]:theory_count[j] for j in range(len(theory_values)) }
    
    unique_values = {*empiric_values, *theory_values}
    freq_empiric = []
    freq_theory = []
    for j in theory_values:
        freq_theory.append( theory_count[j] )
        if j in empiric_count:
            freq_empiric.append( empiric_count[j] )
        else:
            freq_empiric.append( 0 )

    chi_square_test_statistic, p_value = stats.chisquare( freq_empiric, freq_theory, ddof=n-1 ) 
    # chi square test statistic and p value 
    print(f'\u03c7\u00b2 = {chi_square_test_statistic:.3f}') 
    print(f'p_value : {p_value:.3f}') 
    # find Chi-Square critical value 
    print(stats.chi2.ppf(1-0.05, df=6)) 

    plt.plot(quantiles_theory, quantiles_empiric, 'k.', markersize=9, alpha=0.25,)
    plt.axline((1,1), slope=1, alpha=0.5, linewidth=1, c='r')
    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Empirical quantiles')
    plt.xlim([0,max(quantiles_theory)+1])
    plt.ylim([0,max(quantiles_empiric)+1])
    xticks=np.arange( 0, max(quantiles_theory)+2, max(1, max(quantiles_theory)//10) ) 
    plt.xticks(xticks)
    yticks=np.arange( 0, max(quantiles_empiric)+2, max(1, max(quantiles_empiric)//10) ) 
    plt.yticks(yticks)
    plt.grid(alpha=0.75)
    plt.show()
