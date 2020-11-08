# 该文件定义了在树中查找数据所需要的数据结构，类似一个中间件

import copy


# 因为不仅要保存距离，还要存储对应的点，因此设计带index的Distance: DistIndex
class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):
        return self.distance < other.distance


# 用于存储KNN方法中的结果，并更新Worst Distance以更替结果集
# KNNResultSet:
#   capacity：容量即K
#   count：当前大小
#   worst dist：最远距离
#   comparison：比较次数
class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        self.worst_dist = 1e10
        self.dist_index_list = []
        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist, 0))

        self.comparison_counter = 0

    def size(self):
        return self.count

    # 判断是否满了
    def full(self):
        return self.count == self.capacity

    def worstDist(self):
        return self.worst_dist

    # 核心算法：加入点，如果当前点距离不大于worst，将其插入到合适位置，从后向前插入，如果distance小于当前，则将当前位置向后移动，完成后更新最差距离
    # 注意：首先判断是否大于worst，因为初始时距离接近正无穷，所以只要点数>=K就一定会填满，之后再判断是否已满，满了同时更新比较次数

    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.worst_dist:
            return

        if self.count < self.capacity:
            self.count += 1
        i = self.count - 1
        while i > 0:
            if self.dist_index_list[i - 1].distance > dist:
                self.dist_index_list[i] = copy.deepcopy(self.dist_index_list[i - 1])
                i -= 1
            else:
                break

        self.dist_index_list[i].distance = dist
        self.dist_index_list[i].index = index
        self.worst_dist = self.dist_index_list[self.capacity - 1].distance

    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d comparison operations.' % self.comparison_counter
        return output


class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.count = 0
        self.worst_dist = radius
        self.dist_index_list = []

        self.comparison_counter = 0

    def size(self):
        return self.count

    #返回最差距离，这里是定值
    def worstDist(self):
        return self.radius

    #加入判断算法更加简单，只要在半径内就加入结果集
    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.radius:
            return

        self.count += 1
        self.dist_index_list.append(DistIndex(dist, index))

    def __str__(self):
        self.dist_index_list.sort()
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d neighbors within %f.\nThere are %d comparison operations.' \
                  % (self.count, self.radius, self.comparison_counter)
        return output
