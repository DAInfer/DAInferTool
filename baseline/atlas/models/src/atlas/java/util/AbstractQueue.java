package atlas.java.util;
public abstract class AbstractQueue<E> extends atlas.atlas.java.util.AbstractCollection<E> implements atlas.java.util.Queue<E> {
	public AbstractQueue() {
	}
	public boolean add(E p0) {
		boolean r = false;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)p0;
		return (boolean)r;
	}
	public E remove() {
		java.lang.Object r = null;
		r = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4;
		r = (java.lang.Object)((atlas.java.util.PriorityQueue)this).f11;
		r = (java.lang.Object)((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42;
		r = (java.lang.Object)((atlas.java.util.concurrent.LinkedTransferQueue)this).f12;
		r = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedQueue)this).f14;
		return (E)r;
	}
	public void clear() {
	}
	public boolean addAll(atlas.java.util.Collection<? extends E> p0) {
		boolean r = false;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.HashSet)p0).f44;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.ArrayList)p0).f27;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Arrays.ArrayList)p0).f99;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedDeque)p0).f7;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedSet)p0).f73;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Collections.SingletonList)p0).f150;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Collections.SingletonSet)p0).f147;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.LinkedTransferQueue)p0).f12;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Stack)p0).f31;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArraySet)p0).f26;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.PriorityBlockingQueue)p0).f42;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentSkipListSet)p0).f10;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.LinkedHashSet)p0).f18;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.Stack)p0).f31;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Vector)p0).f36;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.Collections.UnmodifiableCollection)p0).f52;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingDeque)p0).f15;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedQueue)p0).f14;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedRandomAccessList)p0).f53;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingQueue)p0).f4;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Collections.UnmodifiableRandomAccessList)p0).f66;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.LinkedList)p0).f19;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingDeque)p0).f15;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.ArrayDeque)p0).f1;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.ArrayDeque)p0).f1;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.PriorityQueue)p0).f11;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.PriorityBlockingQueue)p0).f42;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.HashSet)p0).f44;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingQueue)p0).f4;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.PriorityQueue)p0).f11;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.Vector)p0).f36;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArrayList)p0).f9;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.Vector)p0).f36;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedSet)p0).f73;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedCollection)p0).f2;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.PriorityBlockingQueue)p0).f42;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedQueue)p0).f14;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.LinkedList)p0).f19;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Collections.UnmodifiableCollection)p0).f52;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.LinkedList)p0).f19;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.HashSet)p0).f44;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.IdentityHashMap.KeySet)p0).f165;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.LinkedTransferQueue)p0).f12;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Collections.UnmodifiableSet)p0).f8;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedSet)p0).f73;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingQueue)p0).f4;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.LinkedHashSet)p0).f18;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.LinkedHashSet)p0).f18;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedSortedSet)p0).f70;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.TreeSet)p0).f28;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.ArrayList)p0).f27;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.HashMap.KeySet)p0).f78;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.ArrayDeque)p0).f1;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.Collections.UnmodifiableCollection)p0).f52;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArrayList)p0).f9;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Collections.SingletonList)p0).f150;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedCollection)p0).f2;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Collections.SingletonSet)p0).f147;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedDeque)p0).f7;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArraySet)p0).f26;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingQueue)p0).f4;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Vector)p0).f36;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentSkipListSet)p0).f10;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedCollection)p0).f2;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArrayList)p0).f9;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.LinkedList)p0).f19;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArraySet)p0).f26;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingDeque)p0).f15;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentSkipListSet)p0).f10;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.LinkedHashSet)p0).f18;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingDeque)p0).f15;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingQueue)p0).f4;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.Collections.SingletonList)p0).f150;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.LinkedList)p0).f19;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.Collections.UnmodifiableSortedSet)p0).f169;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.LinkedTransferQueue)p0).f12;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.HashSet)p0).f44;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.PriorityBlockingQueue)p0).f42;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.PriorityQueue)p0).f11;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentSkipListSet)p0).f10;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.PriorityBlockingQueue)p0).f42;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.LinkedTransferQueue)p0).f12;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedDeque)p0).f7;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArraySet)p0).f26;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedSet)p0).f73;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArrayList)p0).f9;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.Vector)p0).f36;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.Stack)p0).f31;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.PriorityQueue)p0).f11;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.ArrayDeque)p0).f1;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.Collections.SingletonList)p0).f150;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.ArrayList)p0).f27;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.Collections.SingletonSet)p0).f147;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedQueue)p0).f14;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.ArrayDeque)p0).f1;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.TreeSet)p0).f28;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.ArrayList)p0).f27;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.TreeSet)p0).f28;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingDeque)p0).f15;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.Collections.SingletonSet)p0).f147;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentSkipListSet)p0).f10;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.PriorityQueue)p0).f11;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedCollection)p0).f2;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Stack)p0).f31;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedDeque)p0).f7;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArraySet)p0).f26;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.Stack)p0).f31;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.HashMap.Values)p0).f38;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.TreeSet)p0).f28;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.concurrent.LinkedTransferQueue)p0).f12;
		((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42 = (java.lang.Object)((atlas.java.util.ArrayList)p0).f27;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.CopyOnWriteArrayList)p0).f9;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.Collections.SynchronizedCollection)p0).f2;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedQueue)p0).f14;
		((atlas.java.util.concurrent.LinkedTransferQueue)this).f12 = (java.lang.Object)((atlas.java.util.TreeSet)p0).f28;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.Collections.SingletonSet)p0).f147;
		((atlas.java.util.PriorityQueue)this).f11 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedQueue)p0).f14;
		((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4 = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedDeque)p0).f7;
		((atlas.java.util.concurrent.LinkedBlockingDeque)this).f15 = (java.lang.Object)((atlas.java.util.LinkedHashSet)p0).f18;
		return (boolean)r;
	}
	public E element() {
		java.lang.Object r = null;
		r = (java.lang.Object)((atlas.java.util.concurrent.LinkedBlockingQueue)this).f4;
		r = (java.lang.Object)((atlas.java.util.PriorityQueue)this).f11;
		r = (java.lang.Object)((atlas.java.util.concurrent.PriorityBlockingQueue)this).f42;
		r = (java.lang.Object)((atlas.java.util.concurrent.LinkedTransferQueue)this).f12;
		r = (java.lang.Object)((atlas.java.util.concurrent.ConcurrentLinkedQueue)this).f14;
		return (E)r;
	}
}
