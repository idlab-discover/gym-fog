class Node(object):
    def __init__(self, simulation=None, number_users=None, next_node=None):
        self.simulation = simulation
        self.number_users = number_users
        self.next = next_node

    def get_sim(self):
        return self.simulation


class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def add(self, sim, number_users):
        node = Node(sim, number_users)
        if self.head is None:
            self.head = node
        else:
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = node

    def get_size(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.next
        return count

    def search(self, number_users):
        current = self.head
        found = False
        while current and found is False:
            if current.get_sim().NUM_USER_REQUESTS == number_users:
                found = True
            else:
                current = current.next
        if current is None:
            raise ValueError("Simulation is not in list")
        return current

    def delete(self, number_users):
        current = self.head
        previous = None
        found = False
        while current and found is False:
            if current.get_sim().NUM_USER_REQUESTS == number_users:
                found = True
            else:
                previous = current
                current = current.get_next()
        if current is None:
            raise ValueError("Simulation is not in the list")
        if previous is None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())

    def traverse_list(self):
        if self.head is None:
            print("List has no element")
            return
        else:
            node = self.head
            while node is not None:
                print(node.number_users, " ")
                node = node.next
