#ifndef _LIST_H_
#define _LIST_H_

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _node {
	void *val;
	struct _node *next;
} node;

typedef struct {
	node *head;
	node *tail;
	int size;
} list;

list *make_list();
void *list_alloc(size_t size);
int list_add_tail(list *l, void *val);
void list_clear(list *l);

#ifdef __cplusplus
}
#endif

#endif