#include <cpprest/http_client.h>
#include <iostream>

using namespace web::http;
using namespace web::http::client;

int main() {
  // Define your Cytoscape server information
  const std::string cytoscapeHost =
      "http://localhost:1234";  // Replace with actual server address

  // Create a Cytoscape node
  http_client client(cytoscapeHost);
  uri_builder builder(U("/v1/networks"));
  http_response response =
      client
          .request(methods::POST, builder.to_string(), U(R"(
        {
            "data": {
                "name": "GeometricGraph"
            }
        }
    )"))
          .get();

  // Check the response status and get the network ID
  if (response.status_code() != status_codes::OK) {
    std::cerr << "Failed to create network." << std::endl;
    return 1;
  }

  std::string networkId = response.to_string();
  networkId = networkId.substr(1, networkId.size() - 2);

  // Create nodes and edges
  builder.set_path(
      U("/v1/networks/") + utility::conversions::to_string_t(networkId) +
      U("/elements"));

  // Create nodes
  client
      .request(methods::POST, builder.to_string(), U(R"(
        [
            {
                "data": {
                    "id": "node1",
                    "label": "Node 1"
                }
            },
            {
                "data": {
                    "id": "node2",
                    "label": "Node 2"
                }
            }
        ]
    )"))
      .wait();

  // Create edges
  client
      .request(methods::POST, builder.to_string(), U(R"(
        [
            {
                "data": {
                    "id": "edge1",
                    "source": "node1",
                    "target": "node2"
                }
            }
        ]
    )"))
      .wait();

  std::cout << "Graph visualization created in Cytoscape." << std::endl;

  return 0;
}
