/* ML hw1 */
// by cysun 0416045

#include <iostream>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cmath>
//#include <random>
#include <iomanip>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#pragma warning(disable:4996)

using namespace std;
#define K 5
#define UNVALID 10000
#define F 4

enum Attribute
{
	sep_l, sep_w, pet_l, pet_w
};



struct Range
{
	double top;
	double bottom;
};


struct Node
{
	vector<Node*> child;
	int split;
	vector<Range> range;
	string class_name;
};

struct Iris
{
	double attr[4];
	/*	sepal_l;
	double sepal_w;
	double petal_l;
	double petal_w;*/
	string  c_name;
};


struct Condition
{
	int attr;
	bool less_great;
};

/* cmp for sorting */
int cmp_sep_l(const Iris &a, const Iris &b)
{
	return a.attr[sep_l] < b.attr[sep_l];
}

int cmp_sep_w(const Iris &a, const Iris &b)
{
	return a.attr[sep_w] < b.attr[sep_w];
}

int cmp_pet_l(const Iris &a, const Iris &b)
{
	return a.attr[pet_l] < b.attr[pet_l];
}

int cmp_pet_w(const Iris &a, const Iris &b)
{
	return a.attr[pet_w] < b.attr[pet_w];
}

// random generator function:
int myrandom(int i) { return std::rand() % i; }

Iris tokenize_line(string l)
{
	Iris cur_iris;
	char line[50];
	char * pch;

	// string to char[]
	strcpy(line, l.c_str());

	stringstream ss0, ss1, ss2, ss3, ss4;

	// sepal_length
	pch = strtok(line, ",");
	ss0 << pch;
	// ss0 >> cur_iris.sepal_l;
	ss0 >> cur_iris.attr[sep_l];

	// sepal_width
	pch = strtok(NULL, ",");
	ss1 << pch;
	//ss1 >> cur_iris.sepal_w;
	ss1 >> cur_iris.attr[sep_w];

	// petal_l
	pch = strtok(NULL, ",");
	ss2 << pch;
	//ss2 >> cur_iris.petal_l;
	ss2 >> cur_iris.attr[pet_l];

	// petal_l
	pch = strtok(NULL, ",");
	ss3 << pch;
	//ss3 >> cur_iris.petal_w;
	ss3 >> cur_iris.attr[pet_w];

	// class name
	pch = strtok(NULL, ",");
	pch = strtok(pch, "-");
	pch = strtok(NULL, "-");

	ss4 << pch;
	ss4 >> cur_iris.c_name;

	return cur_iris;
}

double entropy(vector<Iris> iris)
{
	// probability
	// vector<double> prob;
	// vector<int> appear;
	// vector<string> class_name;
	double p_setosa, p_versicolor, p_virginica;
	double log_se, log_ve, log_vi;
	double ent;

	int s_n = 0, ve_n = 0, vi_n = 0;
	int d_num = iris.size();

	if (iris.empty())
		return 0;

	for (int i = 0; i < iris.size(); i++)
	{
		if (iris[i].c_name == "setosa")
			s_n++;
		else if (iris[i].c_name == "versicolor")
			ve_n++;
		else if (iris[i].c_name == "virginica")
			vi_n++;
	}

	p_setosa = (double)s_n / d_num;
	p_versicolor = (double)ve_n / d_num;
	p_virginica = (double)vi_n / d_num;

	// log
	log_se = (s_n != 0)? log2(p_setosa) : 0;
	log_ve = (ve_n != 0)? log2(p_versicolor) : 0;
	log_vi = (vi_n != 0)? log2(p_virginica) : 0;

	// entropy
	ent = -((p_setosa*log_se) + (p_versicolor*log_ve) + (p_virginica*log_vi));
	return ent;
}

double cal_ent(int count, int d_num)
{
	return (count == 0)? 0 : -(((double)count / d_num) * log2((double)count / d_num));
}

double mid(vector<Iris> iris, int a_type)
{
	// sort by attr
	switch (a_type)
	{
	case sep_l:
		sort(iris.begin(), iris.end(), cmp_sep_l);
		break;
	case sep_w:
		sort(iris.begin(), iris.end(), cmp_sep_w);
		break;
	case pet_l:
		sort(iris.begin(), iris.end(), cmp_pet_l);
		break;
	case pet_w:
		sort(iris.begin(), iris.end(), cmp_pet_w);
		break;
	default:
		break;
	}
	
	int m = iris.size() / 2;
	double mid = iris[m].attr[a_type];
	return mid;
}

bool homo_class(vector<Iris> iris)
{
	bool se = false, ve = false, vi = false;
	for (int i = 0; i < iris.size(); i++)
	{
		if (iris[i].c_name == "setosa")
			se = true;
		else if (iris[i].c_name == "versicolor")
			ve = true;
		else if (iris[i].c_name == "virginica")
			vi = true;
	}

	int tmp = se + ve + vi;
	return (tmp == 1);
}

vector<vector<Iris> > split_data(vector<Iris> iris, int a_type, vector<vector<double> > threshold)
{
	vector<vector<Iris> > level;

	for (int l = 0; l < threshold[a_type].size() -1; l++)
	{
		vector<Iris> part;
		for (int i = 0; i < iris.size(); i++)
		{
			if (iris[i].attr[a_type] >= threshold[a_type][l] && iris[i].attr[a_type] < threshold[a_type][l + 1])
				part.push_back(iris[i]);
		}
		level.push_back(part);
	}

	return level;
}

double rem_buffer[3];

double remainder(vector<Iris> iris, int a_type, vector<vector<double> > threshold)
{
	/* split by features */
	vector<vector<Iris> > level;
	level = split_data(iris, a_type, threshold);

	/* entropy */
	int d_num = iris.size();
	double rem = 0;
	
	for (int l = 0; l < level.size(); l++)
	{
		rem += ((double)level[l].size() / d_num) * entropy(level[l]);
	}

	rem_buffer[2] = rem_buffer[1];
	rem_buffer[1] = rem_buffer[0];
	rem_buffer[0] = rem;
	
	if (rem_buffer[2] == rem_buffer[1] && rem_buffer[1] == rem_buffer[0])
		return UNVALID;
	else
		return rem;
}

Node* build_decision_tree(vector<Iris> iris, vector<vector<double> > threshold)
{
	/* check if finish */
	if (iris.empty())
	{
		Node* node = new Node;
		node->split = 5;
		node->class_name = "empty";
		return node;
	}

	if (homo_class(iris))
	{
		Node* node = new Node;
		node->split = 5;
		node->class_name = (iris[0].c_name == "setosa") ? "setosa" : (iris[0].c_name == "versicolor") ? "versicolor" : "virginica";

		return node;
	}
	
	/* find largest information gain */
	// cur_ent
	double cur_ent, rem, ig, max_ig = 0;
	double sec_ig;
	int cond = -1, sec_cond = 0;
	cur_ent = entropy(iris);
	
	for (int i = sep_l; i <= pet_w; i++)
	{
		rem = remainder(iris, i, threshold);
		ig = cur_ent - rem;
		if (ig >= max_ig)
		{
			max_ig = ig;
			sec_cond = cond;
			cond = i;
		}
	}

	if (cond == -1)
	{
		Node* node = new Node;
		node->split = 5;
		node->class_name = (iris[0].c_name == "setosa") ? "setosa" : (iris[0].c_name == "versicolor") ? "versicolor" : "virginica";

		return node;
	}

	/* split */
	vector<vector<Iris> > level, alt_level;
	vector<Node*> child;
	vector<Range> range;
	bool error = false;

	
	for (int l = 0; l < level.size(); l++)
	{
		if (level[l].size() == iris.size())
			error = true;
	}
	
	if(~error)
		level = split_data(iris, cond, threshold);
	else
		alt_level = split_data(iris, sec_cond, threshold);

	for (int i = 0; i < level.size(); i++)
	{
		Node* c;
		c = build_decision_tree(level[i], threshold);
		child.push_back(c);
	}

	for (int i = 0; i < threshold[cond].size() - 1; i++)
	{
		Range r;
		r.bottom = threshold[cond][i];
		r.top = threshold[cond][i + 1];
		range.push_back(r);
	}

	/*
	if (~error)
	{
		for (int i = 0; i < level.size(); i++)
		{
			Node* c;
			c = build_decision_tree(level[i], threshold);
			child.push_back(c);
		}

		for (int i = 0; i < threshold[cond].size() - 1; i++)
		{
			Range r;
			r.bottom = threshold[cond][i];
			r.top = threshold[cond][i + 1];
			range.push_back(r);
		}
	}
	else
	{
		for (int i = 0; i < alt_level.size(); i++)
		{
			Node* c;
			c = build_decision_tree(alt_level[i], threshold);
			child.push_back(c);
		}

		for (int i = 0; i < threshold[sec_cond].size() - 1; i++)
		{
			Range r;
			r.bottom = threshold[sec_cond][i];
			r.top = threshold[sec_cond][i + 1];
			range.push_back(r);
		}
	}
	*/
	

	/* save cur_node */
	Node* node = new Node;
	node->child = child;
	node->split = cond;
	node->range = range;
	node->class_name = "node";
	return node;
}

void tree_traversal(Node* node, int dep)
{
	dep++;
	if (node == NULL)
	{
		delete node;
		return;
	}

	/*
	cout << "class: " << node->class_name << "  spilt: ";
	string s = (node->split == sep_l) ? "sep_l" : (node->split == sep_w) ? "sep_w" : (node->split == pet_l) ? "pet_l" : (node->split == pet_w)? "pet_w" : "leaf";
	cout << s << ", child = " << node->child.size() << ",  dep = " << dep << endl;
	*/
	
	for (int i = 0; i < node->child.size(); i++)
	{
		tree_traversal(node->child[i], dep);
	}

	delete node;
	return;
}

string classify(Iris cur_iris, Node* tree)
{
	int depth = 0;
	while (tree->class_name == "node")
	{
		
		for (int i = 0; i < tree->child.size(); i++)
		{
			Node* parent = tree;
			if (cur_iris.attr[tree->split] < tree->range[i].top && cur_iris.attr[tree->split] >= tree->range[i].bottom)
			{
				tree = tree->child[i];
				depth++;
				if (tree->class_name == "empty")
					tree = parent->child[i - 1];
				
				break;
			}
		}

		if (tree->class_name == "setosa" || tree->class_name == "versicolor" || tree->class_name == "virginica")
			break;

	}
	
	
	//cout << "classify in " << depth << " times" << endl;

	// finish
	return tree->class_name;
}

void print_iris_vector(vector<Iris> v)
{
	for (int i = 0; i < v.size(); i++)
	{
		cout << v[i].attr[0] << ',' << v[i].attr[1] << ',' << v[i].attr[2] << ',' << v[i].attr[3];
		cout << ',' << v[i].c_name << endl;
	}

	return;
}

void print_iris(Iris i)
{
	cout << i.attr[0] << ',' << i.attr[1] << ',' << i.attr[2] << ',' << i.attr[3];
	cout << ',' << i.c_name << endl;
}

vector<double> feature(vector<Iris> iris, int a_type)
{
	// sort by attr
	switch (a_type)
	{
	case sep_l:
		sort(iris.begin(), iris.end(), cmp_sep_l);
		break;
	case sep_w:
		sort(iris.begin(), iris.end(), cmp_sep_w);
		break;
	case pet_l:
		sort(iris.begin(), iris.end(), cmp_pet_l);
		break;
	case pet_w:
		sort(iris.begin(), iris.end(), cmp_pet_w);
		break;
	default:
		break;
	}

	// pick up features
	//print_iris_vector(iris);

	vector<double> boundry;
	vector<double> thr;
	string name = iris[0].c_name;
	boundry.push_back(0);
	for (int i = 0; i < iris.size(); i++)
	{
		//print_iris(iris[i]);
		if (iris[i].c_name != name)// class change
		{
			//cout << "-----" << endl;
			name = iris[i].c_name;

			bool exist = false;
			for (int b = 0; b < boundry.size(); b++)
			{
				if (boundry[b] == iris[i].attr[a_type])
					exist = true;
			}

			if(!exist)
				boundry.push_back(iris[i].attr[a_type]);
		}
	}

	thr.push_back(0);

	for (int i = 0; i < boundry.size()-1; i++)
	{
		bool exist = false;
		double cur = ((double)boundry[i] + boundry[i + 1]) / 2;
		thr.push_back(cur);
	}

	thr.push_back(UNVALID);

	return thr;
}

int main()
{
	/* open files */
	fstream in_p, out_p;
	in_p.open("iris.data", ios::in | ios::binary);
	
	string l;
	vector<Iris> iris;
	int d_num = 0; // data num

	while (in_p >> l)
	{
		Iris cur_iris;
		cur_iris = tokenize_line(l);
		iris.push_back(cur_iris);	// push into vector
		d_num++;
	}

	in_p.close();

	// shuffle
	
	srand(unsigned(std::time(0)));
	random_shuffle(iris.begin(), iris.end());

	//print_iris_vector(iris);

	/* k fold */
	vector<Iris> se, vi, ve;
	for (int i = 0; i < iris.size(); i++)
	{
		if (iris[i].c_name == "setosa")
			se.push_back(iris[i]);
		else if (iris[i].c_name == "versicolor")
			ve.push_back(iris[i]);
		else if (iris[i].c_name == "virginica")
			vi.push_back(iris[i]);
	}

	// separate into five parts
	vector<Iris> iris_fold[K];

	
	for (int i = 0; i < se.size(); i++)
		iris_fold[i % 5].push_back(se[i]);
	
	for (int i = 0; i < ve.size(); i++)
		iris_fold[i % 5].push_back(ve[i]);

	for (int i = 0; i < vi.size(); i++)
		iris_fold[i % 5].push_back(vi[i]);
	
	// loop k-fold training
	vector<double> se_precision, ve_precision, vi_precision;
	vector<double> se_recall, ve_recall, vi_recall;
	vector<double> accuracy;
	for (int i = 0; i < K; i++)
	{
		// pick up test data
		vector<Iris> test_data;
		test_data = iris_fold[i];

		// merge training data
		vector<Iris> training;
		for (int k = 0; k < K; k++)
		{
			if (k != i)
			{
				for (int d = 0; d < iris_fold[k].size(); d++)
				{
					training.push_back(iris_fold[k][d]);
				}
			}
		}
			
		// set classify features
		vector<vector<double> > threshold;
		threshold.push_back(feature(training, sep_l));
		threshold.push_back(feature(training, sep_w));
		threshold.push_back(feature(training, pet_l));
		threshold.push_back(feature(training, pet_w));

		/* bulid decision tree */
		Node *dTree;
		dTree = build_decision_tree(training, threshold);
		//dep = 0;
		//tree_traversal(dTree, 0);
		
		/* test */
		// int false_negatives = 0, true_positive = 0, false_postives = 0, true_negatives = 0;
		int se_fn = 0, se_tp = 0, se_fp = 0, se_tn = 0;
		int ve_fn = 0, ve_tp = 0, ve_fp = 0, ve_tn = 0;
		int vi_fn = 0, vi_tp = 0, vi_fp = 0, vi_tn = 0;
		int correct = 0;

		// print_iris_vector(test_data);

		for (int t = 0; t < test_data.size(); t++)
		{	
			string data, predict;
			data = test_data[t].c_name;
			test_data[t].c_name = "unknown";
			predict = classify(test_data[t], dTree);

			/*
			for (int y = 0; y < training.size(); y++)
			{
				bool diff = false;
				Iris tst = test_data[t];
				Iris tra = training[y];
				if (tst.c_name == tra.c_name && tst.attr[0] == tra.attr[0] && tst.attr[1] == tra.attr[1] && tst.attr[2] == tra.attr[2] && tst.attr[3] == tra.attr[3])
					cout << "test_data is in the training!!!" << endl;
			}
			*/
			// node->class_name = (se) ? "setosa" : (ve) ? "versicolor" : "virginica";

			correct += (predict == data) ? 1 : 0;

			se_tp += (predict == "setosa" && data == "setosa") ? 1 : 0;
			se_fn += (predict != "setosa" && data == "setosa") ? 1 : 0;
			se_fp += (predict == "setosa" && data != "setosa") ? 1 : 0;
			se_tn += (predict != "setosa" && data != "setosa") ? 1 : 0;

			ve_tp += (predict == "versicolor" && data == "versicolor") ? 1 : 0;
			ve_fn += (predict != "versicolor" && data == "versicolor") ? 1 : 0;
			ve_fp += (predict == "versicolor" && data != "versicolor") ? 1 : 0;
			ve_tn += (predict != "versicolor" && data != "versicolor") ? 1 : 0;

			vi_tp += (predict == "virginica" && data == "virginica") ? 1 : 0;
			vi_fn += (predict != "virginica" && data == "virginica") ? 1 : 0;
			vi_fp += (predict == "virginica" && data != "virginica") ? 1 : 0;
			vi_tn += (predict != "virginica" && data != "virginica") ? 1 : 0;
		}

		
		// precision & recall
		double se_prec, ve_prec, vi_prec;
		double se_rc, ve_rc, vi_rc;

		se_prec = (double)se_tp / (se_tp + se_fp);
		ve_prec = (double)ve_tp / (ve_tp + ve_fp);
		vi_prec = (double)vi_tp / (vi_tp + vi_fp);

		se_precision.push_back(se_prec);
		ve_precision.push_back(ve_prec);
		vi_precision.push_back(vi_prec);

		se_rc = (double)se_tp / (se_fn + se_tp);
		ve_rc = (double)ve_tp / (ve_fn + ve_tp);
		vi_rc = (double)vi_tp / (vi_fn + vi_tp);
		
		se_recall.push_back(se_rc);
		ve_recall.push_back(ve_rc);
		vi_recall.push_back(vi_rc);

		// accuracy
		double acc;

		acc = (double)correct / test_data.size();
		accuracy.push_back(acc);

		// delete nodes
		tree_traversal(dTree, 0);
	}

	
	double se_avg_prec = 0, ve_avg_prec = 0, vi_avg_prec = 0;
	double se_avg_rc = 0, ve_avg_rc = 0, vi_avg_rc = 0;
	double avg_acc = 0;

	for (int i = 0; i < K; i++)
	{
		se_avg_prec += se_precision[i];
		ve_avg_prec += ve_precision[i];
		vi_avg_prec += vi_precision[i];
		se_avg_rc += se_recall[i];
		ve_avg_rc += ve_recall[i];
		vi_avg_rc += vi_recall[i];
		avg_acc += accuracy[i];
	}
	
	se_avg_prec = (double)se_avg_prec / K;
	ve_avg_prec = (double)ve_avg_prec / K;
	vi_avg_prec = (double)vi_avg_prec / K;
	se_avg_rc = (double)se_avg_rc / K;
	ve_avg_rc = (double)ve_avg_rc / K;
	vi_avg_rc = (double)vi_avg_rc / K;
	avg_acc = (double)avg_acc / K;
	
	cout << fixed << setprecision(3) << avg_acc << endl;
	cout << fixed << setprecision(3) << se_avg_prec << ' ' << se_avg_rc << endl;
	cout << fixed << setprecision(3) << ve_avg_prec << ' ' << ve_avg_rc << endl;
	cout << fixed << setprecision(3) << vi_avg_prec << ' ' << vi_avg_rc << endl;

	return 0;
}