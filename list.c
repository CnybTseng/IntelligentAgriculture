#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list()
{
	list *l = calloc(1, sizeof(list));
	if (!l) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return l;
	}
	
	l->head = NULL;
	l->tail = NULL;
	l->size = 0;
	
	return l;
}

void *list_alloc(size_t size)
{
	return  calloc(1, size);
}

int list_add_tail(list *l, void *val)
{
	if (!l) {
		fprintf(stderr, "invalid list[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	node *n = calloc(1, sizeof(node));
	if (!n) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	n->val  = val;
	n->next = NULL;
	
	if (!l->head) {
		l->head = n;
		l->tail = n;
	} else {
		l->tail->next = n;
		l->tail = n;
	}
	++l->size;
	
	return 0;
}

void list_clear(list *l)
{
	if (!l) return;
	
	node *n = l->head;
	while (n) {
		node *item = n;
		n = n->next;
		if (item->val) free(item->val);
		free(item);
	}
	
	free(l);
}