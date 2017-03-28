
import sys; 




def importTest(str1):

	sys.path.insert(0, './testFolder/')

	test = __import__(str1);   

	className = 'TestClass'; 


	test = __import__(str1, globals(), locals(), [className],0); 
	tester = test.TestClass; 

	t = tester(5); 

	t.p(); 




if __name__ == "__main__":
	

	if(len(sys.argv) == 1):
		print('Input Import statement'); 
		imp = raw_input(); 

	else:
		imp = sys.argv[1]; 

	importTest(imp); 

