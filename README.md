# What
A system to easily create a "system of prompting"  
Final project will contain a UI for making system of prompting  
## Design
Start with making CLI to create a simple system. Must define expected input and output. Then define how many LLM passes to make, and create templates for each pass. Harder part might be structured output.
If a sub prompt is something like:  
Given X, Tell us Y and Z. Then Y and Z are used independantly in other prompts.
Different possibilities: depending on the response of a subprompt, we may want to trigger different flows.   
Overall, the input and output will behave like a current chatbot, but the underlying system will consist of multiple llm prompts and flows depending on outputs that will eventually reach a more accurate response.
## 
User can:
- "+" add nodes in a graph
- "-" remove nodes from a graph
- Inspect and edit templates for nodes
- When adding a node, must decide where the input comes from, and where the output goes
- Removing node must deal with the same issues
- There must be an overall input and an overall output of the entire system
- 

What is a node in the graph?
Node is a instruction + prompt for an llm. It has some input, either absolute input or from another node. This input is assembled into a prompt along with instructions. The prompt assembly is decided by some template.
Nodes are in some way just an automatic prompting of an llm (currently api only) where the inputs are either user provided (at start of system) or automatically passed from a previous node.


Testing:
Depending on the systems goals we can write tests, and even test individual nodes within a system. Actually, tests should be encorporated as an important part of the project. Its easy and useless to design a simple prompt system framework, but its much more useful to easily design and test it.   
Chess example:


- give a chess move
- Is this a valid move
- Depending on answer if no, circle back to start with provided context.

We can test the "is this a valid move" part independently by passing moves that we know are valid or not.

Create some examples:
- Simple agent example?
- Chess example
- Advanced social media example





NODE:

TEMPLATE:  

CONNECTION:

INPUT:

OUTPUT:


