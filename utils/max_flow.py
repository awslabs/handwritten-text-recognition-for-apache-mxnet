# source: https://brilliant.org/wiki/ford-fulkerson-algorithm/

# def get_max_flow_graph(prob):
#     fn = FlowNetwork()
#     fn.add_vertex('source')
#     fn.add_vertex('drain')
#     for i in range(prob.shape[0]):
#         for j in range(prob.shape[1]):
#             vertex = str(i) + "_" + str(j)
#             fn.add_vertex(vertex)
#             if i > 0:
#                 for k in range(prob.shape[1]):
#                     previous_vertex = str(i - 1) + "_" + str(k)
#                     fn.add_edge(previous_vertex, vertex, w=prob[k, i])
#             else:
#                 fn.add_edge("source", vertex, w=prob[j, i])
                
#     for i in range(prob.shape[1]):
#         fn.add_edge(str(prob.shape[0] - 1)+"_" + str(i), 'drain', w=0)
#     flow = fn.max_flow(source="source", sink="drain")

# def max_flow(prob):
#     '''
#     Uses a maximum flow network to obtain multiple possible candidates for the sentence.
#     1) Parse each word based spaces (if max probability is a space)
#     2) Build a max flow network for each word
#     3) Obtain candidates for each word and compare them to a spell checker.
#     '''
#     word_probs = []
#     for i in range(prob.shape[1]):
#         prob_i = prob[0, i, :]
#         if np.argmax(prob_i) == alphabet_dict[" "]:
#             probs = np.array(word_probs)
#             get_max_flow_graph(probs)
#         word_probs.append(prob_i)
        
#     predicted_text = topK_decode(np.argmax(prob, axis=2))[0]
    
#     output = ""
#     for word in predicted_text.split(" "):
#         output += simple_spellcheck(word) + " "
#     return output

class Edge(object):  
    def __init__(self, u, v, w):
        self.source = u
        self.sink = v
        self.capacity = w

    def __repr__(self):
        return "%s->%s:%s" % (self.source, self.sink, self.capacity)


class FlowNetwork(object):  
    def __init__(self):
        self.adj = {}
        self.flow = {}

    def add_vertex(self, vertex):
        self.adj[vertex] = []

    def get_edges(self, v):
        return self.adj[v]

    def add_edge(self, u, v, w=0):
        if u == v:
            raise ValueError("u == v")
        edge = Edge(u, v, w)
        redge = Edge(v, u, 0)
        edge.redge = redge
        redge.redge = edge
        self.adj[u].append(edge)
        self.adj[v].append(redge)
        self.flow[edge] = 0
        self.flow[redge] = 0

    def find_path(self, source, sink, path):
        if source == sink:
            return path
        for edge in self.get_edges(source):
            residual = edge.capacity - self.flow[edge]
            if residual > 0 and edge not in path:
                result = self.find_path(edge.sink, sink, path + [edge])
                if result is not None:
                    return result

    def max_flow(self, source, sink):
        path = self.find_path(source, sink, [])
        while path is not None:
            residuals = [edge.capacity - self.flow[edge] for edge in path]
            flow = min(residuals)
            for edge in path:
                self.flow[edge] += flow
                self.flow[edge.redge] -= flow
            path = self.find_path(source, sink, [])
        return sum(self.flow[edge] for edge in self.get_edges(source))
